import numpy as np
import mujoco
import weakref
import collections
import copy


_INVALID_JOINT_NAMES_TYPE = (
    "`joint_names` must be either None, a list, a tuple, or a numpy array; " "got {}."
)

_REQUIRE_TARGET_POS_OR_QUAT = (
    "At least one of `target_pos` or `target_quat` must be specified."
)

IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


class IKSolver:
    def __init__(
        self,
        model_ref: weakref.ref,
        data_ref: weakref.ref,
        max_steps=100,
        tolerance=1e-6,
    ):
        self._m = model_ref
        self._d = data_ref
        self._max_steps = max_steps
        self._max_update_norm = 2.0
        self._progress_thresh = 20.0
        self._tolerance = tolerance

    @property
    def m(self):
        return self._m()

    @property
    def d(self):
        return self._d()

    def qpos_from_site_xpos(
        self,
        site_name,
        target_pos=None,
        target_quat=None,
        joint_names=None,
    ):
        dtype = self.d.qpos.dtype
        data = self.d

        if target_pos is not None and target_quat is not None:
            jac = np.empty((6, self.m.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jacp, jacr = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, self.m.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if target_pos is not None:
                jacp, jacr = jac, None
                err_pos, err_rot = err, None
            elif target_quat is not None:
                jacp, jacr = None, jac
                err_pos, err_rot = None, err
            else:
                raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

        update_nv = np.zeros(self.m.nv, dtype=dtype)

        if target_quat is not None:
            site_xquat = np.empty(4, dtype=dtype)
            neg_site_xquat = np.empty(4, dtype=dtype)
            err_rot_quat = np.empty(4, dtype=dtype)

        # Ensure that the Cartesian position of the site is up to date.
        mujoco.mj_fwdPosition(self.m, data)

        site_xpos = data.site(site_name).xpos
        site_xmat = data.site(site_name).xmat
        site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site_name)

        # This is an index into the rows of `update` and the columns of `jac`
        # that selects DOFs associated with joints that we are allowed to manipulate.
        if joint_names is None:
            dof_indices = slice(None)  # Update all DOFs.
        elif isinstance(joint_names, (list, np.ndarray, tuple)):
            if isinstance(joint_names, tuple):
                joint_names = list(joint_names)
            # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
            # indexer to map each joint name to the indices of its corresponding DOFs.
            indexer = [self.m.joint(jn).jntid for jn in joint_names]
            dof_indices = np.asarray(indexer).flatten()
        else:
            raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

        steps = 0
        success = False

        for steps in range(self._max_steps):
            err_norm = 0.0

            if target_pos is not None:
                # Translational error.
                err_pos[:] = target_pos - site_xpos
                err_norm += np.linalg.norm(err_pos)
            if target_quat is not None:
                # Rotational error.
                mujoco.mju_mat2Quat(site_xquat, site_xmat)
                mujoco.mju_negQuat(neg_site_xquat, site_xquat)
                mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
                mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
                err_norm += np.linalg.norm(err_rot) * 0.1

            if err_norm < self._tolerance:
                success = True
                break
            else:
                mujoco.mj_jacSite(self.m, data, jacp, jacr, site_id)
                jac_joints = jac[:, dof_indices]

                # TODO Account for joint limits.
                reg_strength = 3e-2 if err_norm > 0.1 else 0.0
                update_joints = nullspace_method(
                    jac_joints, err, regularization_strength=reg_strength
                )

                update_norm = np.linalg.norm(update_joints)

            progress_criterion = err_norm / update_norm
            if progress_criterion > self._progress_thresh:
                break

            if update_norm > self._max_update_norm:
                update_joints *= self._max_update_norm / update_norm
            
            update_nv[dof_indices] = update_joints

            # update_nv[dof_indices] = update_joints
            mujoco.mj_integratePos(self.m, data.qpos, update_nv, 1)
            mujoco.mj_fwdPosition(self.m, data)

        qpos = data.qpos

        return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.
    Args:
      jac_joints: The Jacobian of the end effector with respect to the joints. A
        numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
        and `nv` is the number of degrees of freedom.
      delta: The desired end-effector delta. A numpy array of shape `(3,)` or
        `(6,)` containing either position deltas, rotation deltas, or both.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements. Default is zero, i.e. no regularization.
    Returns:
      An `(nv,)` numpy array of joint velocities.
    Reference:
      Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
      transpose, pseudoinverse and damped least squares methods.
      https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
