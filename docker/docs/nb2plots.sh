#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Regenerate the README.rst file from the top level README
pandoc --from=markdown --to=rst --output=README.rst $DIR/../README.md

pushd $DIR

rm -rf jupyter
mkdir -p jupyter/figures

jupyter_notebook_paths=(
"nxsdk/tutorials/ipython/a_compartment_setup.ipynb"
"nxsdk/tutorials/ipython/b_connecting_compartments.ipynb"
"nxsdk/tutorials/ipython/c_stimulating_compartments.ipynb"
"nxsdk/tutorials/ipython/d_synaptic_plasticity.ipynb"
"nxsdk/tutorials/ipython/e_neuronal_homeostasis.ipynb"
"nxsdk/tutorials/ipython/f_rewards_learning.ipynb"
"nxsdk/tutorials/ipython/g_snip_for_threshold_modulation/g_snip_for_threshold_modulation.ipynb"
"nxsdk/tutorials/ipython/h_reward_trace.ipynb"
"nxsdk/tutorials/ipython/i_performance_profiling.ipynb"
"nxsdk/tutorials/ipython/j_soft_reset_core.ipynb"
"nxsdk/tutorials/ipython/j_soft_reset_net.ipynb"
"nxsdk/tutorials/ipython/k_interactive_spike_sender_receiver.ipynb"
"nxsdk/tutorials/ipython/l_snip_for_compartment_setup_with_NxNet_C/l_snip_for_compartment_setup_with_NxNet_C.ipynb"
"nxsdk/tutorials/ipython/m_independent_networks_in_same_run.ipynb"
"nxsdk/tutorials/ipython/n_stochastic_networks.ipynb"
"nxsdk/tutorials/ipython/o_snip_for_reading_lakemont_spike_count/o_snip_for_reading_lakemont_spike_count.ipynb"
"nxsdk/tutorials/ipython/p_composable_networks.ipynb"
"nxsdk/tutorials/ipython/q_connection_sharing.ipynb"
"nxsdk/tutorials/ipython/r_stubs_and_netmodules.ipynb"
"nxsdk/tutorials/ipython/s_multicx_neuron_self_reward.ipynb"
"nxsdk/tutorials/ipython/t_vMinExp_and_vMaxExp.ipynb"
"nxsdk/tutorials/ipython/u_join_op_in_multi_compartment_neuron.ipynb"
"nxsdk_modules/lca/tutorials/a_solving_convolutional_lasso_with_lca.ipynb"
"nxsdk_modules/slic/tutorials/single_layer_image_classifier.ipynb"
"nxsdk_modules/epl/tutorials/epl_multi_pattern_oneshot_learning.ipynb"
"nxsdk_modules/dnn/tutorials/a_image_classification_mnist.ipynb"
"nxsdk_modules/dnn/tutorials/b_image_classification_cifar.ipynb"
"nxsdk_modules/path_planning/tutorials/tutorial_path_planning.ipynb"
"nxsdk_modules/characterization/tutorials/timeAndEnergyBarrierSync.ipynb"
"nxsdk_modules/characterization/tutorials/timeAndEnergyPerOp.ipynb"
"nxsdk_modules/knn/tutorials/knnGist.ipynb"
)

# Copy notebooks
for i in ${jupyter_notebook_paths[@]};
do
    cmd="cp -v ../$i jupyter"
    echo $cmd
    $cmd
done

# Copy figures
rm -rf figures
mkdir figures

figure_dirs=(
"nxsdk/tutorials/ipython/figures"
"nxsdk_modules/dnn/tutorials/figures"
"nxsdk_modules/slic/tutorials/figures"
"nxsdk_modules/path_planning/tutorials/figures"
"nxsdk_modules/characterization/tutorials/figures"
)

for directory in ${figure_dirs[@]};
do
    cp ../$directory/* jupyter/figures/
done

popd
