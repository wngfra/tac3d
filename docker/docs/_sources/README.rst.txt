NxSDK Release History
=====================

1.0.0
-----

--------------

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

General
^^^^^^^

-  lakemont_driver has been re-named to nx
-  Timestamps for time and energy probes are collected on x86 cores
   between tStart and tEnd, as indicated in the probe condition,
   reducing the number of times the x86 and host must sync.
-  Boot-up times accelerated by initializing only used chips and
   powering down the unused cores.
-  Energy probes on Kapoho Bay now include start and end temperatures
   (all timesteps in between will return NaN to indicate no valid
   temerature reading)
-  Fixed a bug where the number of unique axonal delays per core was
   artificially limited by NxSDK
-  The location of N2Board has changed. It must now be imported from
   nxsdk.arch.n2a.n2board, or through the N2A API as
   nxsdk.api.n2a.NxBoard
-  The syntax board.n2Chips[].n2Cores[] is still supported for now, but
   will give a deprecation warning because the syntax is changing to
   board.nxChips[].nxCores[] to make changing between chip and core
   versions easier

NxSDK Modules
^^^^^^^^^^^^^

-  NxSlayer auto modules: Release of NxSlayer auto modules that enable
   seamless network creation and execution of SLAYER-Loihi trained
   models in Loihi hardware. The auto modules are available as
   ``nxSlayer.auto.{s2lDataset, Network}``
-  NxSlayer benchmarking module: Utililty to create multiple copies of
   NxSlayer auto network and replicate it over multiple Loihi chips for
   accurate energy benchmarking. It is available as
   ``nxSlayer.benchmark.MultiChipNetwork``

Known Issues
~~~~~~~~~~~~

.. _section-1:

0.9.9
-----

--------------

.. _new-featuresimprovements-1:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-1:

General
^^^^^^^

-  Power Telemetry measurement for Kapoho Bay is now supported

.. _nxsdk-modules-1:

NxSDK Modules
^^^^^^^^^^^^^

-  KNN : Release of the approximate nearest neighbor module accompanying
   the paper: E. Paxon Frady, Garrick Orchard, David Florey, Nabil Imam,
   Ruokun Liu, Joyesh Mishra, Jonathan Tse, Andreas Wild, Friedrich T.
   Sommer, and Mike Davies. 2020. Neuromorphic Nearest Neighbor Search
   Using Intel’s Pohoiki Springs. In Proceedings of the Neuro-inspired
   Computational Elements Workshop (NICE ’20). Association for Computing
   Machinery, New York, NY, USA, Article 23, 1–10.
   DOI:https://doi.org/10.1145/3381755.3398695

.. _known-issues-1:

Known Issues
~~~~~~~~~~~~

-  The frequency at which energy measurements are obtained on Kapoho Bay
   is significantly lower than for Nahuku32, so some workloads need to
   be run for more timesteps to get enough energy measurements for
   automatic calculation of dynamic power consumption

.. _section-2:

0.9.8
-----

--------------

.. _new-featuresimprovements-2:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-2:

General
^^^^^^^

-  NxSDK Support for Pohoiki Springs
-  Power Telemetry measurement for Pohoiki Springs
-  When using energy probes on Pohoiki Springs, LMT option platform-args
   must be set to specify the column to probe. This can be set through
   the command line arguments via
   ``LMTOPTIONS=[--platform-args="<relative_column>/<total_columns>/<absolute_column>"]``
   or through board.options. For example
   ``LMTOPTIONS=[--platform-args="0/1/0"]`` ,
   ``LMTOPTIONS=[--platform-args="0/1/1"]``, and
   ``LMTOPTIONS=[--platform-args="0/1/2"]`` will probe the leftmost
   column, middle column and rightmost column, respectively.
-  Added support for all python versions by removing the pyc
   precompilation step during pip packaging

NxNet
^^^^^

-  enableDelay attributed in Connection and ConnectionPrototype has been
   replaced by disableDelay. If disableDelay is set to True, then delay
   is not used as a synaptic delay but a general purpose variable for
   learning only.
-  Added compiler support for synaptic tag variables to allow
   specification of initial tag values.

.. _nxsdk-modules-2:

NxSDK Modules
^^^^^^^^^^^^^

-  Spike Input Generator module has been released for the Composable
   Framework (can be found in the Github repo)
-  Spike Input Streamer provides a means to inject spikes into a model
   at deterministic times while the model is running on Loihi.

NxCore
^^^^^^

-  A new snip phase ``EMBEDDED_REMOTE_MGMT`` allows code to be remotely
   triggered by pre-emption.
-  Upper limit of user channels which can be created has been increased
   to 1024

.. _known-issues-2:

Known Issues
~~~~~~~~~~~~

-  Energy probes for Pohoiki Springs will not be recording temperature.
   This feature will be removed across all hardware in future as it adds
   no meaningful insight.
-  Static power measurement across Pohoiki Springs boards should be
   ignored for now.

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  DVS module’s addSpikes() function now correctly configures x86 cores
   to receive spikes from the host instead of expecting a live hardware
   interface from DAVIS240C.

.. _section-3:

0.9.5
-----

--------------

.. _new-featuresimprovements-3:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-3:

General
^^^^^^^

-  Introduced a flag in /etc/nx/board.conf to hint whether a board
   supports power measurement or not (“supportsPowerMeasurement”:“false”
   or “true”). If this flag does not exist, the default behaviour is
   true.
-  Few modules such as EPL, DNN and Loihi backend for SNNToolbox have
   been moved out to external github
   https://github.com/intel-nrc-ecosystem.
-  Optimized barrier sync algorithm which will lead to reduction in time
   per time step and lower energy consumption in some cases.

.. _nxnet-1:

NxNet
^^^^^

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| u_join_op_in_multi- | Compartment        | This tutorial | Multi-co |
| compartment_neuron. | settings           | demonstrates  | mpartmen |
| ipynb               |                    | the different | t        |
|                     |                    | joint         | neuron   |
|                     |                    | operations in |          |
|                     |                    | a             |          |
|                     |                    | multi-compart |          |
|                     |                    | ment          |          |
|                     |                    | neuron.       |          |
+---------------------+--------------------+---------------+----------+

.. _nxsdk-modules-3:

NxSDK Modules
^^^^^^^^^^^^^

-  Documentation for the SNN-Toolbox backend for Loihi (See NxSDK
   Modules -> DNN).
-  Several bug fixes of the SNN-Toolbox backend and NxTF.
-  Overhauled ANN to SNN conversion now supporting different activation
   behaviors (reset-to-zero, reset-by-subtraction, saturation).
   Especially reset-by-subtraction allows for higher accuracy.
-  Complete integration of InputGenerator with NxModel for faster input
   injection.
-  New end to end tutorial for CIFAR-10 from network setup and training
   to conversion with the SNN-Toolbox and fast inference with the
   InputGenerator.

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| a_image_classificat | DNN                | Demonstrate   | DNN      |
| ion_mnist.ipynb     |                    | running DNN   |          |
|                     |                    | and           |          |
|                     |                    | Composability |          |
|                     |                    | on Loihi on   |          |
|                     |                    | MNIST data    |          |
+---------------------+--------------------+---------------+----------+
| b_image_classificat | DNN                | Demonstrate   | DNN      |
| ion_cifar.ipynb     |                    | running DNN,  |          |
|                     |                    | Composability |          |
|                     |                    | and SNN       |          |
|                     |                    | Toolbox on    |          |
|                     |                    | Loihi on      |          |
|                     |                    | CIFAR10 data  |          |
+---------------------+--------------------+---------------+----------+

.. _nxcore-1:

NxCore
^^^^^^

-  Channels in python now support probeChannel. So users can now probe
   the channel to avoid a blocking read (in case of recv channel) or
   blocking write (in case of send channel).
-  Board object exposes a new attribute accessible via
   ``board.energyTimeMonitor.powerProfileStats``. This is populated once
   board.disconnect is invoked and stores (as dictionary) resource and
   power utilization of the execution.
-  Execution Time and Energy Probes now use more precise timestamps. So
   using them will increase the memory utilization on lmt 0 chip 0 and
   leave less memory for SNIP scheduled on that specific cpu since this
   release. Other lakemont CPUs are unaffected. If you get linking error
   while compiling snips, please adjust the bin size and buffer size to
   accumulate lesser data.

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  tracecfggen module has moved to nxsdk/arch/n2a/compiler/tracecfggen

.. _known-issues-3:

Known Issues
~~~~~~~~~~~~

-  Messages sent from chips other than chip 0 to superhost on the last
   timestep of the run might not be delivered. The workaround to fix it
   is to run for 1 extra timestep which ensures all messages are flushed
   correctly before the next timestep starts.

.. _major-bug-fixes-for-release-1:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  A deadlock issue related to channel communication due to race
   condition was fixed. It was primarily observed on KapohoBay.

.. _section-4:

0.9.0
-----

--------------

.. _new-featuresimprovements-4:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-4:

General
^^^^^^^

-  **NxTF** : Users should be able to create (or port from existing)
   deep neural networks on Loihi. NxTF implements the Keras interface.
   Users can also use SNN Toolbox to port exist DNNs to Loihi.
-  **Composability** : Provides an interface to enable modules,
   applications and frameworks to be stitched together and compiled down
   to the NxBoard Graph.
-  **Slayer** : Slayer integration has been added within nxsdk_modules.
   It allows to run models pre-trained using slayer on Loihi
-  **Embedded Execution Engine** : A new execution engine to run on
   embedded platform with minimum memory requirements
-  Concurrent Host Snips are now available for users to run concurrent
   execution on host while neurocore execution is in progress
-  DVS :

   -  Performance improvement - Access to the snip for users to modify
      the spike injection

-  Basic Spike Generator : Optimized the spike gen (5-10x improvement)

   -  Users can specify which LMT it will run on (removing the
      requirement that they are on the same LMT in same phase)

-  Dependencies for nxsdk_modules have been moved out of nxsdk into its
   own requirements.txt. Hence, you might need to optionally run
   ``pip install -r nxsdk-apps/nxsdk_modules/requirements.txt`` to fetch
   dependencies for some of these modules.
-  Optimized barrier sync algorithm which will lead to reduction in time
   per time step and lower energy consumption in some cases.

.. _nxnet-2:

NxNet
^^^^^

-  NxSDK compiler now supports user specified custom partitioners and
   mappers. A custom partitioner and mapper can be individually
   specified for each network in a network tree.
-  Updated documentation for APIs to use and query resource maps

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| s_multicx_neuron_se | Mutlicompartment   | This tutorial | NxNet    |
| lf_reward.ipynb     | neurons, reward    | illustrates   |          |
|                     | channels           | learning a    |          |
|                     |                    | synaptic      |          |
|                     |                    | weight to     |          |
|                     |                    | synchronise   |          |
|                     |                    | spiking of    |          |
|                     |                    | soma and      |          |
|                     |                    | dendrite      |          |
|                     |                    | compartments  |          |
+---------------------+--------------------+---------------+----------+
| t_vMinExp_and_vMaxE | Compartment        | This tutorial | NxNet    |
| xp.ipynb            | settings           | demonstrates  |          |
|                     |                    | the usage of  |          |
|                     |                    | vMaxExp and   |          |
|                     |                    | vMinExp in    |          |
|                     |                    | compartments  |          |
+---------------------+--------------------+---------------+----------+

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_26_concur | Host Snips         | This tutorial | Snips    |
| rent_host_snips_wi |                    | demonstrates  |          |
| th_yarp_integratio |                    | using yarp    |          |
| n                  |                    | ports with    |          |
|                    |                    | concurrent    |          |
|                    |                    | host snips.   |          |
+--------------------+--------------------+---------------+----------+

.. _nxsdk-modules-4:

NxSDK Modules
^^^^^^^^^^^^^

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| nxsdk_modules/dnn/t | DNN mapping        | This tutorial | NxTF     |
| utorials/a_nxtf_par |                    | demonstrates  |          |
| titioning_mapping   |                    | how to use    |          |
|                     |                    | the NxTF      |          |
|                     |                    | framework to  |          |
|                     |                    | partition a   |          |
|                     |                    | DNN for       |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+
| nxsdk_modules/dnn/t | DNN mapping        | This tutorial | NxTF     |
| utorials/b_nxtf_mni |                    | demonstrates  |          |
| st                  |                    | how to use    |          |
|                     |                    | the NxTF      |          |
|                     |                    | framework to  |          |
|                     |                    | run a DNN on  |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+
| nxsdk_modules/dnn/t | DNN mapping        | This tutorial | NxTF     |
| utorials/c_snntb_mn |                    | demonstrates  |          |
| ist                 |                    | how to use    |          |
|                     |                    | the SNN       |          |
|                     |                    | Toolbox to    |          |
|                     |                    | run a DNN on  |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+
| nxsdk_modules/dnn/t | Composables        | This tutorial | NxTF     |
| utorials/d_mnist_us |                    | demonstrates  |          |
| ing_composables     |                    | how to use    |          |
|                     |                    | the           |          |
|                     |                    | composability |          |
|                     |                    | interface to  |          |
|                     |                    | compose DNN   |          |
|                     |                    | with Input    |          |
|                     |                    | Generator and |          |
|                     |                    | run the MNIST |          |
|                     |                    | network on    |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+
| nxsdk_modules/slaye | SLAYER fully       | This tutorial | SLAYER   |
| r/tutorials/nmnist/ | connected          | demonstrates  |          |
| NMNIST              |                    | how to run a  |          |
|                     |                    | SLAYER-traine |          |
|                     |                    | d             |          |
|                     |                    | fully         |          |
|                     |                    | connected     |          |
|                     |                    | model on      |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+
| nxsdk_modules/slaye | SLAYER convnet     | This tutorial | SLAYER   |
| r/tutorials/gesture |                    | demonstrates  |          |
| /gesture            |                    | how to run a  |          |
|                     |                    | SLAYER-traine |          |
|                     |                    | d             |          |
|                     |                    | convolutional |          |
|                     |                    | model on      |          |
|                     |                    | Loihi.        |          |
+---------------------+--------------------+---------------+----------+

.. _nxcore-2:

NxCore
^^^^^^

-  **Embedded Execution**: Use board.generateEEEArtifacts() to package
   and distribute execution artifact to be run in an embedded
   environment. This is primarily useful when the only interaction is
   via host snips (All communication and execution happen between host
   and neurocores/lmt)
-  Serialization/Deserialization performance has been improved by using
   binary format. This makes board.dumpNeuroCores(…) and
   board.dumpNeuroCores(…) faster

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_22_host_s | Host Snips         | This tutorial | Snips    |
| nips_with_yarp_int |                    | demonstrates  |          |
| egration           |                    | using yarp    |          |
|                    |                    | ports with    |          |
|                    |                    | host snips.   |          |
+--------------------+--------------------+---------------+----------+
| tutorial_23_host_s | Host Snips         | This tutorial | Snips    |
| nips_with_ros_inte |                    | demonstrates  |          |
| gration            |                    | using ros     |          |
|                    |                    | with host     |          |
|                    |                    | snips.        |          |
+--------------------+--------------------+---------------+----------+
| tutorial_24_contro | Snips              | This tutorial | Snips    |
| l_loop_using_rospy |                    | demonstrates  |          |
|                    |                    | using ros     |          |
|                    |                    | with channels |          |
|                    |                    | from          |          |
|                    |                    | superhost     |          |
|                    |                    | useful for    |          |
|                    |                    | proto-typing  |          |
|                    |                    | with ROS.     |          |
+--------------------+--------------------+---------------+----------+

.. _api-changesdeprecations-1:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  Custom partitioners that partitioned an entire network tree is no
   longer supported. Custom partitioners are now specified on a per
   NxNet object level.
-  In createChannel : elementType would be deprecated in 0.9 in favor of
   messageSize, which provides more flexibility of specifying size of a
   message.

.. _major-bug-fixes-for-release-2:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Fixed the slowness issue while loading neurocores
   (board.loadNeuroCores) on KapohoBay
-  Fixed a bug related to logical id spilling over core boundary

.. _section-5:

0.8.7
-----

--------------

.. _new-featuresimprovements-5:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-3:

NxNet
^^^^^

-  ``epl`` : EPL module now works for multi-pattern learning and has an
   updated Jupyter notebook tutorial which shows how the EPL network can
   learn and recall not only odors but also images. EPL is a
   neuromorphic one-shot learning algorithm which can learn and recall
   patterns using a spatio-temporal attractor network inspired by the
   neural circuitry of the external plexiform layer (EPL) of the
   mammalian olfactory bulb.

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_25_sequen | Host Snips         | This tutorial | Snips    |
| tial_host_snips    |                    | demonstrates  |          |
|                    |                    | how to setup  |          |
|                    |                    | sequential    |          |
|                    |                    | host snips,   |          |
|                    |                    | provide       |          |
|                    |                    | implementatio |          |
|                    |                    | n             |          |
|                    |                    | and           |          |
|                    |                    | scheduling    |          |
|                    |                    | and connect   |          |
|                    |                    | them to       |          |
|                    |                    | embedded      |          |
|                    |                    | snips via     |          |
|                    |                    | channels      |          |
+--------------------+--------------------+---------------+----------+

.. _nxcore-3:

NxCore
^^^^^^

-  N2Board and N2Chip now expose various methods for allocating chips
   and cores on demand (not only during N2Board setup) at manually or
   automatically determined mesh locations. Manually determine mesh
   locations can be specified by (x, y, p) address or logical core
   index.

.. _general-5:

General
^^^^^^^

-  Board supports creation of host snips in sequential and concurrent
   mode. See documentation for further detail on host snips api.
-  Improved live DVS support. Users can now write their own snips to
   receive live DVS spikes and inject them into their model.

.. _api-changesdeprecations-2:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  board.createProcess API is being deprecated. Use board.createSnip
   instead.
-  board.n2Chips[k].n2Cores now returns a dict mapping the logical core
   index to the n2Core object instead of a list of n2Core objects. This
   is to support the ability to allocate cores at arbitrary mesh
   locations. A list of allocated n2Core objects can be accessed through
   board.n2Chips[k].n2CoresAsList.
-  net.createDVSSpikeGenProcess API is being deprecated. Use the DVS
   module instead. See the DVS module tutorial for an example.

.. _major-bug-fixes-for-release-3:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _section-6:

0.8.5.1
-------

--------------

.. _new-featuresimprovements-6:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxcore-4:

NxCore
^^^^^^

-  Synapse sharing between discrete axons

.. _section-7:

0.8.5
-----

--------------

.. _new-featuresimprovements-7:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-4:

NxNet
^^^^^

-  Composable networks
-  Connection sharing (i.e. synapse sharing)
-  Following modules have been added under nxsdk_modules (Each module
   comes with tutorials to demonstrate applications or re-usable feature
   set)

   -  ``noise filter`` : Performs noise filtering on a DVS stream.
   -  ``dvs`` : Provides an interface to DVS sensors. Currently only
      supports the DAVIS240C.
   -  ``trace injection`` : A module showing how non-local information
      can be injected into traces available to the learning rules.
   -  ``path planning`` : Path planning model is based on the planning
      algorithm modeled after the operational principles of hippocampal
      place cells. The algorithm infers associations between neurons in
      a network from the asymmetric effects of STDP on a propagating
      sequence of spikes.
   -  ``epl`` : A neuromorphic one-shot learning algorithm which can
      learn and recall patterns using a spatio-temporal attractor
      network inspired by the neural circuitry of the external plexiform
      layer (EPL) of the mammalian olfactory bulb.

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| p_composable_networ | Composable         | This tutorial | NxNet    |
| ks.ipynb            | networks           | illustrates   |          |
|                     |                    | how to        |          |
|                     |                    | connect       |          |
|                     |                    | multiple      |          |
|                     |                    | NxNet objects |          |
+---------------------+--------------------+---------------+----------+
| q_connection_sharin | Connection sharing | This tutorial | NxNet    |
| g.ipynb             |                    | illustrates   |          |
|                     |                    | the use of    |          |
|                     |                    | shared        |          |
|                     |                    | connections   |          |
+---------------------+--------------------+---------------+----------+
| r_stubs_and_netmodu | Stubs and          | This tutorial | NxNet    |
| les.ipynb           | NetModules         | illustrates   |          |
|                     |                    | the use of    |          |
|                     |                    | connection    |          |
|                     |                    | stubs and     |          |
|                     |                    | making a      |          |
|                     |                    | module        |          |
+---------------------+--------------------+---------------+----------+

.. _nxcore-5:

NxCore
^^^^^^

-  Axon Compiler now supports remote population axons
-  BasicSpikeGenerator now supports pop16 and pop32 spike type

.. _general-6:

General
^^^^^^^

-  Several improvements in transfer performance (Network/IO) should
   speed up data transfer. Speedup has been observed in configuring
   chips/registers and probe post-processing.
-  Speed and memory use for creating large compartment groups and
   connection groups have both been significantly improved.
-  SpikeOutputPorts allow fast communication of spikes from Loihi to a
   pipe on the host when using Kapoho Bay
-  Improved DVS support for Kapoho Bay
-  Support for custom partitioners
-  Setup instructions have been updated for KapohoBay and boards which
   are maintained outside INRC. Please revisit getting started guides

.. _api-changesdeprecations-3:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  NxNet.createConnection(src, dst, …) and
   NxNet.createConnectionGroup(src, dst, …) are no longer supported. The
   supported way to create connections is src.connect(dst, …).
-  nx.ConnectionGroup(…), nx.CompartmentGroup(…), and nx.NeuronGroup(…)
   have been deprecated. The supported method for creating these groups
   is net.createConnectionGroup(…), net.createCompartmentGroup(…), and
   net.createNeuronGroup(…)
-  Spike Probes will only start accumulating spikes only after tstart
   has been configured and any previous spikes are discarded.
-  Synapse recompilation is not supported. Re-encoding should be done
   via SNIPs which provide a faster way for re-configuration.
-  The group id property (for example, CompartmentGroup.id) has been
   renamed to groupId (i.e. CompartmentGroup.groupId) to avoid confusion
   with python id() function.
-  nxDriver is deprecated as a property of Graph/N2Board. Instead please
   use board.executor.
-  startDriver is being deprecated. Instead please use board.start.
-  Spike Generators have been moved within nxsdk.graph.nxinputgen
   module. For e.g. to import BasicSpikeGenerator you would **import
   nxsdk.graph.nxinputgen.nxinputgen.BasicSpikeGenerator**.
-  Spike Receivers will now only send incremental data since the last
   invocation.
-  board.dump() and board.load() have been renamed to
   board.dumpNeuroCores() and board.loadNeuroCores() respectively.
-  The DVS noise filter interface has changed. See the corresponding
   tutorials for examples of how to use it.

.. _major-bug-fixes-for-release-4:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Add proper handling when a compartment prototype is used multiple
   times in a compartment prototype tree (neuron API)

.. _section-8:

0.8.1
-----

--------------

.. _new-featuresimprovements-8:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-5:

NxNet
^^^^^

-  Compilation and network creation optimizations

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| o_snip_for_reading_ | spikes count       | This tutorial | SNIP     |
| lakemont_spike_coun |                    | illustrates   |          |
| t.ipynb             |                    | the use of    |          |
|                     |                    | spike probes  |          |
|                     |                    | to capture    |          |
|                     |                    | the spikes    |          |
|                     |                    | count from    |          |
|                     |                    | lakemont via  |          |
|                     |                    | SNIP          |          |
+---------------------+--------------------+---------------+----------+

.. _general-7:

General
^^^^^^^

-  Added test_pio.bin to test defects on parallel IO (PIO) links and
   validate chip-to-chip PIO links
-  Improved DVS integration

.. _section-9:

0.8.0
-----

--------------

.. _new-featuresimprovements-9:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-6:

NxNet
^^^^^

-  Spike Receivers are enabled in NxNet for asynchronous and interactive
   probing of spikes
-  Enhancements were added to performance framework to report phase wise
   distribution including energy metrics and new workloads and
   categories have been added
-  Logging support is now enabled within NxSDK
-  Compiler optimizations to shorten compilation time
-  Addition of the neuron API to simplify the construction of
   multi-compartment neurons. See Neuron and NeuronPrototype
   documentation for details
-  Energy probes are now enabled on all platforms (WM, Nahuku) except
   KapohoBay (USB)
-  Improved stability and speedup of NxSDK on KapohoBay (USB)

+---------------------+--------------------+---------------+----------+
| Jupyter Tutorials   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| i_performance_profi | Performance        | This tutorial | Probes   |
| ling.ipynb          | Profiling          | demonstrates  |          |
|                     | (Execution Time    | how to        |          |
|                     | and Energy)        | measure the   |          |
|                     |                    | performance   |          |
|                     |                    | of your       |          |
|                     |                    | networks from |          |
|                     |                    | the           |          |
|                     |                    | perspective   |          |
|                     |                    | of time,      |          |
|                     |                    | energy and    |          |
|                     |                    | power         |          |
|                     |                    | utilization   |          |
|                     |                    | as well as    |          |
|                     |                    | phase-wise    |          |
|                     |                    | distribution  |          |
+---------------------+--------------------+---------------+----------+
| j_soft_reset_net.ip | neuron API         | This tutorial | NxNet    |
| ynb                 |                    | demonstrates  |          |
|                     |                    | soft reset    |          |
|                     |                    | membrane      |          |
|                     |                    | voltage at    |          |
|                     |                    | NxNet level   |          |
|                     |                    | using neuron  |          |
|                     |                    | API           |          |
+---------------------+--------------------+---------------+----------+
| k_interactive_spike | Interactive        | This tutorial | Spike    |
| _sender_receiver.ip | SpikeGen and Spike | demonstrates  | Injectio |
| ynb                 | Receiver           | how to set up | n,       |
|                     |                    | spike         | Probes   |
|                     |                    | generators    |          |
|                     |                    | (basic and    |          |
|                     |                    | interactive)  |          |
|                     |                    | which enable  |          |
|                     |                    | to send       |          |
|                     |                    | stimuli both  |          |
|                     |                    | before and    |          |
|                     |                    | during the    |          |
|                     |                    | run. It also  |          |
|                     |                    | highlights    |          |
|                     |                    | the use of    |          |
|                     |                    | Spike         |          |
|                     |                    | Receivers to  |          |
|                     |                    | receive       |          |
|                     |                    | spikes in a   |          |
|                     |                    | non-blocking  |          |
|                     |                    | manner during |          |
|                     |                    | the run       |          |
+---------------------+--------------------+---------------+----------+
| l_snip_for_compartm | NxNet C API        | This tutorial | SNIP     |
| ent_setup_with_NxNe |                    | demonstrates  |          |
| t_C.ipynb           |                    | how to use    |          |
|                     |                    | NxNet C API   |          |
|                     |                    | to setup      |          |
|                     |                    | compartments  |          |
+---------------------+--------------------+---------------+----------+
| m_independent_netwo | networks           | This tutorial | NxNet    |
| rks_in_same_run.ipy |                    | demonstrates  |          |
| nb                  |                    | that several  |          |
|                     |                    | independent   |          |
|                     |                    | and           |          |
|                     |                    | disconnected  |          |
|                     |                    | networks can  |          |
|                     |                    | be configured |          |
|                     |                    | in the same   |          |
|                     |                    | run           |          |
+---------------------+--------------------+---------------+----------+

.. _nxcore-6:

NxCore
^^^^^^

-  Snips can now run on any lakemont on any chip provided, number of
   chips allocated > chipId for the snip.
-  Added Tutorial 21 demonstrating the multi-lmt snips api.

.. _major-bug-fixes-for-release-5:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Deadlock arising from incorrect packing of synapses when learning is
   enabled has been fixed. This fix was a regression in 0.7.5 where a
   fix was put in to syn pack/unpack mechanism.
-  Memory leaks in probes has been fixed on host
-  Fixed a problem where, depending on the compartment ordering,
   compartment voltage may have non-zero decay even though voltage decay
   is set to 0.
-  Issue with using the spike generator to send spikes selectively
   across multiple net.run invocations has been fixed.

.. _known-issues-4:

Known Issues
~~~~~~~~~~~~

-  In certain circumstances, compressionMode = DENSE can cause undefined
   behavior. We recommend not specifying dense mode unless you truly
   need it.
-  Due to limited memory availability on the embedded device (Lakemont),
   allocating large memory segments within SNIPs might result in a
   linking error such as

   .. code:: bash

          collect2: error: ld returned 1 exit status
          /usr/bin/ld: temp/launcher.link section `.data' will not fit in region `DTCM'
          /usr/bin/ld: region `DTCM' overflowed by 472005 bytes

   Reduce the memory utilization or use efficient packing protocols to
   create your data structures. If you are getting these errors while
   using execution time/energy probes, use binning with appropriate bin
   size.

.. _api-changesdeprecations-4:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  NxSDK module layout has been refactored to support new generations of
   hardware. Common code was moved to the base level of the module. Some
   user code might need to modify their imports if they are using these
   sub-modules (nxsdk.arch.n2a.compiler -> nxsdk.compiler,
   nxsdk.arch.n2a.graph -> nxsdk.graph and so on).
-  Removed chipGen parameter from NxCore board.run interface. Chip
   generation is now automatically configured.
-  avoid-deadlock is removed from list of valid compiler options.
-  Learning rule specification using bracket expressions, i.e. (v+C),
   requires expanding out the terms for certain ranges of C values. See
   Specifying a Learning Rule documentation for details.
-  ChipId is now a 3D structure with x,y,z fields, while the id union
   remains 14 bits. This will better support the topology of the
   upcoming Poihiki Springs systems.
-  Various nx_sw_\* and nx_ne_\* functions are renamed to nx_min_\* and
   nx_max_\* for improved clarity with the 3D coordinates.
-  The nx_phys_chipid/nx_phys_coreid functions are deprecated, replaced
   by nx_nth_chipid/nx_nth_coreid. The new functions return ChipId or
   CoreId’s id fields that increase monotonically with n. That is, the
   order in which chips are connected on the FPIO serial chain is now
   hidden. Therefore nx_nth_chipid(0) will always be nx_min_chipid() and
   nx_nth_chipid(nx_num_chips()-1) will always be nx_max_chipid()
   Requesting an out-of-bounds core or chip will exit with an error.
-  runState structure’s fields, time and nsteps have been renamed to
   time_step and total_steps respectively.

.. _section-10:

0.7.5
-----

--------------

.. _new-featuresimprovements-10:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-7:

NxNet
^^^^^

-  Enable “useDiscreteVTh” parameter in compartment prototype. This
   parameter should be enabled if many different threshold voltages are
   set in a given core.
-  Enable “axonDelay” parameter in compartment prototype to set the axon
   delay.
-  Added runAsync and stop (abort) API to NxNet for interactive
   processing.
-  NxNet supports InteractiveSpikeGenProcess to send spikes during
   runtime (post compilation).
-  Added support for saving and loading the state of the board
   (serialization and deserialization) 

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_22_activi | Activity Probe     | This tutorial | Probes   |
| ty_probe           |                    | demonstrates  |          |
|                    |                    | how to set up |          |
|                    |                    | activity      |          |
|                    |                    | probes to     |          |
|                    |                    | monitor       |          |
|                    |                    | spikes. The   |          |
|                    |                    | SDK offers    |          |
|                    |                    | spike probes  |          |
|                    |                    | which execute |          |
|                    |                    | faster and    |          |
|                    |                    | are easier to |          |
|                    |                    | set up but    |          |
|                    |                    | are limited   |          |
|                    |                    | by the        |          |
|                    |                    | available     |          |
|                    |                    | spike         |          |
|                    |                    | counters      |          |
|                    |                    | limited by    |          |
|                    |                    | the number of |          |
|                    |                    | spike         |          |
|                    |                    | counters per  |          |
|                    |                    | chip. Per     |          |
|                    |                    | chip we offer |          |
|                    |                    | around 2400   |          |
|                    |                    | counters. As  |          |
|                    |                    | a workaround  |          |
|                    |                    | for the       |          |
|                    |                    | limited       |          |
|                    |                    | number of     |          |
|                    |                    | spike         |          |
|                    |                    | counters per  |          |
|                    |                    | chip,         |          |
|                    |                    | activity      |          |
|                    |                    | probes can be |          |
|                    |                    | used instead. |          |
|                    |                    | Since the     |          |
|                    |                    | update of the |          |
|                    |                    | compartment   |          |
|                    |                    | activity      |          |
|                    |                    | variable is   |          |
|                    |                    | bound to the  |          |
|                    |                    | threshold     |          |
|                    |                    | homeostasis   |          |
|                    |                    | feature,      |          |
|                    |                    | threshold     |          |
|                    |                    | homeostasis   |          |
|                    |                    | must be       |          |
|                    |                    | enabled. In   |          |
|                    |                    | order to      |          |
|                    |                    | prevent the   |          |
|                    |                    | actual        |          |
|                    |                    | threshold to  |          |
|                    |                    | change the    |          |
|                    |                    | homeostasisGa |          |
|                    |                    | in            |          |
|                    |                    | parameter     |          |
|                    |                    | must be set   |          |
|                    |                    | to zero. In   |          |
|                    |                    | addition,     |          |
|                    |                    | since the     |          |
|                    |                    | activity      |          |
|                    |                    | variable is   |          |
|                    |                    | modeled as a  |          |
|                    |                    | spike trace,  |          |
|                    |                    | its decay     |          |
|                    |                    | time constant |          |
|                    |                    | must be set   |          |
|                    |                    | to a large    |          |
|                    |                    | value for it  |          |
|                    |                    | to act as a   |          |
|                    |                    | spike         |          |
|                    |                    | counter.      |          |
+--------------------+--------------------+---------------+----------+
| tutorial_21_dvs_in | Serialization/De-s | This tutorial | Utility  |
| putfile_with_serde | erialization       | illustrates   |          |
|                    |                    | how to use    |          |
|                    |                    | serialization |          |
|                    |                    | /de-serializa |          |
|                    |                    | tion          |          |
|                    |                    | to skip pass  |          |
|                    |                    | the           |          |
|                    |                    | compilation   |          |
|                    |                    | phase by      |          |
|                    |                    | saving the    |          |
|                    |                    | state of the  |          |
|                    |                    | board during  |          |
|                    |                    | a successful  |          |
|                    |                    | run and       |          |
|                    |                    | reload it the |          |
|                    |                    | later run.    |          |
|                    |                    | DVS tutorial  |          |
|                    |                    | is used to    |          |
|                    |                    | demonstrate   |          |
|                    |                    | this.         |          |
+--------------------+--------------------+---------------+----------+

.. _nxcore-7:

NxCore
^^^^^^

-  Updated NxCore API documentation by adding a section that explains
   the basic object hierachy as well as more detailed register field
   explanations.

-  Add population mode support to axon compiler.  

.. _major-bug-fixes-for-release-6:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  NxNet rewards compilation unexpectedly raises a bounds error when
   more than 4 compartments attempt to share a reward.
-  Synapse compilation error results in unexpected behavior when
   switching between sign modes.

.. _api-changesdeprecations-5:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  In NxNet, the relationship between synaptic “delay” and the
   compartment prototype parameter “numDendriticAccumulators” has been
   updated to properly reflect Loihi behavior. Specifically, for all
   synapses connecting to a given compartment, the maximum synaptic
   “delay” < “numDendriticAccumulators”-1.

.. _section-11:

0.7.1
-----

--------------

.. _major-bug-fixes-for-release-7:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Jupyter tutorial for LCA has been fixed to correctly find the
   associated dictionary and input image dataset
-  Version in requirements.txt and setup.py are not pinned to exact
   versions instead allow a range of versions for python dependencies

.. _known-issues-5:

Known Issues
~~~~~~~~~~~~

-  Networks using more than 2 chips might fail on N2A2 based Kapoho Bay
   which can only support upto 2 Chips at the moment. Such runs might
   hang.

Any new/supported platforms/Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Software support for N2A2 based Kapoho Bay was added

.. _section-12:

0.7.0
-----

--------------

Prerequisites
~~~~~~~~~~~~~

-  python 3.5.2
-  pip
-  jupyter (for running IPython notebooks)

We highly recommend you to create a virtual environment first.

If you are working on INRC cloud, these packages are pre-installed. To
use jupyter within INRC, see **Jupyter Tutorials** section below.

Setup
~~~~~

NxSDK is now packaged as a pip package. The tutorials, documentation and
modules are separately distributed as a tarball.

You may look into the modules to understand how to implement your own
modules or extend them. Follow the instructions below to install NxSDK.

Please Note: The same instructions apply while you test NxSDK on Kapoho
Bay USB device; however as a pre-requisite - you need to download all
the necessary installation bits to your host machine from the INRC
Cloud. Follow the instructions you received with Kapoho Bay.

1. The installation bits are located at \ **/nfs/ncl/releases**. SSH
   into the INRC Cloud and check the contents of this directory:

   -  **ssh user@YOURVM.research.intel-research.net**
   -  **cd /nfs/ncl/releases**

   Identify the version you want to install. It is recommended to
   install the latest version.

2. There should be 2 packages within the <latest_version> sub-directory:

   1. *nxsdk-<latest_version>.tar.gz* - A pip installable tarball to
      install NxSDK
   2. *nxsdk-apps-<latest_version>.tar.gz* - Tarball for Jupyter/Python
      based tutorials, complete NxSDK documentation and NxSDK Modules

      Substitute above your username, VM and version information before
      running the next commands.

3. Create a virtual environment in your home directory:

   -  ``cd ~``
   -  ``python3 -m venv python3_venv``
   -  ``source python3_venv/bin/activate``
   -  ``pip install -U pip``
   -  Do all following steps within the virtual environment

4. Copy Release Artifacts: **cp /nfs/ncl/releases/<latest_version>/\*
   .**

5. Install NxSDK: \ **python -m pip install
   nxsdk-<latest_version>.tar.gz**

   -  Ignore the “Failed building wheel for nxsdk” and the associated
      “Failed to build nxsdk”. This is a known error and pip will retry
      with setup.py.

6. Verify NxSDK installation: \ **python -c “import nxsdk;
   print(nxsdk.__version__)”**
7. Find NxSDK installation directory (nxsdk_install_dir): **python -c
   “import nxsdk; print(nxsdk.__path__)”**

8. Unzip Tutorials, Docs and Modules in your home directory: \ **mkdir
   nxsdk-apps && tar xzf nxsdk-apps-<latest_version>.tar.gz -C
   nxsdk-apps –strip-components 1**

   -  Run tutorials per instructions in the following sections.

9. When complete, deactivate the virtual environment by running
   ``deactivate``

Tutorials
~~~~~~~~~

Jupyter Tutorials (IPython Notebooks)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working within the INRC Cloud, follow instructions below to
run Jupyter

-  Ensure you are in the virtual environment
   (``source python3_venv/bin/activate``)
-  Run \ **SLURM=1 /nfs/ncl/bin/jupyter_nx.sh**
-  Follow the instructions on your terminal screen to enable SSH
   tunneling and browser access and use Jupyter from the web

   -  If the browser asks for a password or token, please copy and paste
      the hex string at the end of the web address into the token field.
   -  For example:

      -  http://localhost:8891/?token=c14eb1f2cee91e5b2fea550e9ca1ce65721c892b4b88dbf1
      -  The token is: c14eb1f2cee91e5b2fea550e9ca1ce65721c892b4b88dbf1

-  Once you can access the jupyter web UI from your browser, locate the
   nxsdk-apps directory and open the IPython notebooks from paths
   mentioned below

Jupyter based tutorials demonstrate NxSDK features and examples using
Python code snippets. Each tutorial focuses on certain feature set of
the NxSDK API.

You will find all foundational jupyter notebooks
under \ **nxsdk-apps/tutorials/ipython**

Jupyter notebooks to demonstrate modules are located under each module.

For example, the jupyter notebook for LCA tutorial is
at \ **nxsdk-apps/nxsdk_modules/lca/tutorials/lca_single_image_reconstruction.ipynb**

Python Tutorials
^^^^^^^^^^^^^^^^

Python based tutorials demonstrate NxSDK features and examples using
Python code snippets.

Each tutorial focuses on certain feature set of the NxSDK API and can be
run both from command line or IDE based tools

``source python3_venv/bin/activate``

``cd nxsdk-apps/tutorials``

To run a NxNet
tutorial, \ ``SLURM=1 python -m nxnet.tutorial_01_single_compartment_with_bias`` 

-  All NxNet tutorials are under \ **nxsdk-apps/tutorials/nxnet**

To run a NxCore
tutorial, \ ``SLURM=1 python -m nxcore.tutorial_01_single_compartment_with_bias`` 

-  All NxCore tutorials are under \ **nxsdk-apps/tutorials/nxcore**

To View Documentation
~~~~~~~~~~~~~~~~~~~~~

``cd nxsdk-apps/docs``

You can start a simple http server or serve the docs via web-server like
Apache or Nginx

``python -m http.server``

You would need to do SSH tunneling to view the docs.

Alternatively, you may download the docs directory to your local machine
(Windows or Mac) and click on the index.html. You can do the same if you
have a VNC session connected to the INRC Cloud.

Release Notes 0.7.0
~~~~~~~~~~~~~~~~~~~

.. _new-featuresimprovements-11:

New Features/Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _nxnet-8:

NxNet
'''''

-  0.7 introduces the NxNet Modules

   -  Modules are higher level abstractions to re-use existing networks
      and create applications on top of them using very few lines of
      code
   -  Modules bundled in this release are:

      -  Spiking Locally Competitive Algorithm (LCA)
      -  Single Layer Image Classification (SLIC)

-  Full support for Kapoho Bay USB device to run NxSDK
-  Support reinforcement on a learning rule using Reinforcement Channels
-  DVS Camera interface to inject spikes into compartments in NxNet
-  Support for execution time probes and energy probes (only for Wolf
   Mountain boards)
-  Several performance speed-ups such as compilation time improvements,
   time-series processing and message aggregations
-  NxNet now supports run and disconnect APIs. Users can call net.run()
   directly instead of using the N2Compiler explicitly

+---------------------+--------------------+---------------+----------+
| Jupyter Notebooks   | Feature Covered    | Description   | Category |
+=====================+====================+===============+==========+
| a_compartment_setup | NxSDK Core         | Within this   | Compartm |
| .ipynb              | Components 1.      | tutorial, you | ents     |
|                     | Basic compartment  | will learn    |          |
|                     | with probes 2.     | how to create |          |
|                     | Compartment with   | a compartment |          |
|                     | compartmentVoltage | neuron,       |          |
|                     | Decay              | configure the |          |
|                     | 3. Compartment     | bias current  |          |
|                     | with (stochastic)  | and monitor   |          |
|                     | compartmentVoltage | the resulting |          |
|                     | 4. Compartment     | spike         |          |
|                     | with (stochastic)  | behavior      |          |
|                     | refractoryDelay    | using probes. |          |
|                     |                    | You will also |          |
|                     |                    | learn to      |          |
|                     |                    | configure     |          |
|                     |                    | voltage       |          |
|                     |                    | decays,       |          |
|                     |                    | refractory    |          |
|                     |                    | delays as     |          |
|                     |                    | well as       |          |
|                     |                    | stochasticity |          |
|                     |                    | in            |          |
|                     |                    | compartment   |          |
|                     |                    | voltage or    |          |
|                     |                    | current due   |          |
|                     |                    | to uniform    |          |
|                     |                    | random noise  |          |
|                     |                    | injection.    |          |
+---------------------+--------------------+---------------+----------+
| b_connecting_compar | Configuring        | The second    | Connecti |
| tments.ipynb        | Connections using  | tutorial      | vity     |
|                     | Connection         | builds on the | and Flow |
|                     | Prototypes and     | first,        |          |
|                     | Connection Groups  | showing you   |          |
|                     | 1. Connecting      | now how to    |          |
|                     | single             | connect       |          |
|                     | compartments 2.    | compartments  |          |
|                     | Connecting         | using         |          |
|                     | compartment groups | connection    |          |
|                     | with discrete      | prototypes    |          |
|                     | weight and delay   | and           |          |
|                     | matrices 3. Create | connections.  |          |
|                     | connections with   | It walk you   |          |
|                     | different sign     | through       |          |
|                     | mode 4. Create     | simple        |          |
|                     | connections with   | connections   |          |
|                     | different post     | to using      |          |
|                     | synaptic response  | features in   |          |
|                     | mode               | the           |          |
|                     |                    | Connection    |          |
|                     |                    | Prototype to  |          |
|                     |                    | support more  |          |
|                     |                    | complex       |          |
|                     |                    | synaptic      |          |
|                     |                    | attributes.   |          |
+---------------------+--------------------+---------------+----------+
| c_stimulating_compa | Stimulate          | The third     | Connecti |
| rtments.ipynb       | compartments by    | tutorial      | vity     |
|                     | injecting spikes   | builds on the | and Flow |
|                     |                    | first two to  |          |
|                     |                    | demonstrate   |          |
|                     |                    | how to        |          |
|                     |                    | stimulate     |          |
|                     |                    | compartments  |          |
|                     |                    | by injecting  |          |
|                     |                    | spikes using  |          |
|                     |                    | a Spike       |          |
|                     |                    | Generator. A  |          |
|                     |                    | spike         |          |
|                     |                    | generator is  |          |
|                     |                    | a class that  |          |
|                     |                    | can either    |          |
|                     |                    | generate      |          |
|                     |                    | spikes based  |          |
|                     |                    | on some       |          |
|                     |                    | algorithm,    |          |
|                     |                    | from a file   |          |
|                     |                    | or represent  |          |
|                     |                    | a hardware    |          |
|                     |                    | sensor such   |          |
|                     |                    | as a DVS      |          |
|                     |                    | camera.       |          |
+---------------------+--------------------+---------------+----------+
| d_synaptic_plastici | Demonstrate        | In this       | Connecti |
| ty.ipynb            | synaptic           | tutorial, the | vity     |
|                     | plasticity to      | network we    | and Flow |
|                     | study the weight   | configure     |          |
|                     | dynamics of a      | consists of a |          |
|                     | synapse subject to | spike         |          |
|                     | an excitatory STDP | generator     |          |
|                     | learning rule      | that          |          |
|                     |                    | stimulates a  |          |
|                     |                    | learning-enab |          |
|                     |                    | led           |          |
|                     |                    | synapse using |          |
|                     |                    | an E-STDP     |          |
|                     |                    | learning      |          |
|                     |                    | rule. We then |          |
|                     |                    | use another   |          |
|                     |                    | spike         |          |
|                     |                    | generator to  |          |
|                     |                    | independently |          |
|                     |                    | drive a       |          |
|                     |                    | compartment   |          |
|                     |                    | that forms a  |          |
|                     |                    | post synaptic |          |
|                     |                    | connection of |          |
|                     |                    | the           |          |
|                     |                    | learning-enab |          |
|                     |                    | led           |          |
|                     |                    | synapse. This |          |
|                     |                    | test setup    |          |
|                     |                    | allows to     |          |
|                     |                    | control the   |          |
|                     |                    | pre and post  |          |
|                     |                    | synaptic      |          |
|                     |                    | spike times   |          |
|                     |                    | at will       |          |
|                     |                    | independently |          |
|                     |                    | and to study  |          |
|                     |                    | the induced   |          |
|                     |                    | weight        |          |
|                     |                    | dynamics.     |          |
+---------------------+--------------------+---------------+----------+
| e_neuronal_homeosta | Demonstrates       | The Loihi     | Compartm |
| sis.ipynb           | Neuronal           | neuron model  | ents     |
|                     | homeostasis to     | supports a    |          |
|                     | adapt the          | built-in      |          |
|                     | excitability of a  | homeostasis   |          |
|                     | neuron to          | rule which we |          |
|                     | different levels   | call \ *range |          |
|                     | of input stimulus  | homeostasis.* |          |
|                     | intensity          | Other         |          |
|                     |                    | mechanisms to |          |
|                     |                    | adapt the     |          |
|                     |                    | excitability  |          |
|                     |                    | of a          |          |
|                     |                    | compartment   |          |
|                     |                    | can be        |          |
|                     |                    | implemented   |          |
|                     |                    | by            |          |
|                     |                    | constructing  |          |
|                     |                    | multi-compart |          |
|                     |                    | ment          |          |
|                     |                    | neurons which |          |
|                     |                    | allows to     |          |
|                     |                    | design more   |          |
|                     |                    | complex       |          |
|                     |                    | neural        |          |
|                     |                    | behaviors. In |          |
|                     |                    | this          |          |
|                     |                    | tutorial, we  |          |
|                     |                    | demonstrate   |          |
|                     |                    | the           |          |
|                     |                    | implementatio |          |
|                     |                    | n             |          |
|                     |                    | of another    |          |
|                     |                    | homeostatic   |          |
|                     |                    | rule which we |          |
|                     |                    | call          |          |
|                     |                    | ‘exponential- |          |
|                     |                    | decay’        |          |
|                     |                    | homeostasis.  |          |
+---------------------+--------------------+---------------+----------+
| f_rewards_learning. | Demonstrates       | Loihi         | Learning |
| ipynb               | Learning with      | provides the  |          |
|                     | Rewards            | capability to |          |
|                     |                    | send          |          |
|                     |                    | reinforcement |          |
|                     |                    | signals to    |          |
|                     |                    | learning      |          |
|                     |                    | enabled       |          |
|                     |                    | connections.  |          |
|                     |                    | This tutorial |          |
|                     |                    | shows how to  |          |
|                     |                    | reinforce the |          |
|                     |                    | contingent    |          |
|                     |                    | firing of two |          |
|                     |                    | neurons by a  |          |
|                     |                    | delayed       |          |
|                     |                    | reward        |          |
|                     |                    | (fundamental  |          |
|                     |                    | mechanism     |          |
|                     |                    | described by  |          |
|                     |                    | Izhikevich et |          |
|                     |                    | al. to solve  |          |
|                     |                    | the distal    |          |
|                     |                    | reward        |          |
|                     |                    | problem)      |          |
+---------------------+--------------------+---------------+----------+
| g_snip_for_threshol | Demonstrates the   | We stimulate  | SNIP     |
| d_modulation.ipynb  | basic use of SNIP  | a single      |          |
|                     |                    | compartment   |          |
|                     |                    | with a spike  |          |
|                     |                    | generator but |          |
|                     |                    | vary the      |          |
|                     |                    | membrane      |          |
|                     |                    | threshold     |          |
|                     |                    | over time via |          |
|                     |                    | a SNIP. We    |          |
|                     |                    | send the new  |          |
|                     |                    | membrane      |          |
|                     |                    | threshold     |          |
|                     |                    | value from    |          |
|                     |                    | the super     |          |
|                     |                    | host to a     |          |
|                     |                    | SNIP          |          |
|                     |                    | executing in  |          |
|                     |                    | the           |          |
|                     |                    | management    |          |
|                     |                    | phase on the  |          |
|                     |                    | embedded      |          |
|                     |                    | Lakemont      |          |
|                     |                    | processor via |          |
|                     |                    | a channel and |          |
|                     |                    | return the    |          |
|                     |                    | time of       |          |
|                     |                    | modification  |          |
|                     |                    | to the super  |          |
|                     |                    | host via      |          |
|                     |                    | another       |          |
|                     |                    | channel       |          |
+---------------------+--------------------+---------------+----------+
| lca_single_image_re | Spiking Locally    | The LCA       | Modules  |
| construction.ipynb  | Competitive        | network       |          |
|                     | Algorithm          | solves a      |          |
|                     |                    | sparse coding |          |
|                     |                    | problem using |          |
|                     |                    | a spiking     |          |
|                     |                    | version of a  |          |
|                     |                    | locally       |          |
|                     |                    | competitive   |          |
|                     |                    | algorithm     |          |
|                     |                    | (LCA). The    |          |
|                     |                    | purpose of    |          |
|                     |                    | this tutorial |          |
|                     |                    | is to         |          |
|                     |                    | illustrate    |          |
|                     |                    | the basic     |          |
|                     |                    | network and   |          |
|                     |                    | setup         |          |
|                     |                    | procedure as  |          |
|                     |                    | well as how   |          |
|                     |                    | to solve a    |          |
|                     |                    | sparse coding |          |
|                     |                    | problem and   |          |
|                     |                    | retrieve the  |          |
|                     |                    | results.      |          |
+---------------------+--------------------+---------------+----------+

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_11_box_sy | Box Synapse        | This tutorial | Connecti |
| napse              |                    | introduces a  | vity     |
|                    |                    | box synapse.  |          |
|                    |                    | Instead of an |          |
|                    |                    | exponentially |          |
|                    |                    | decaying post |          |
|                    |                    | synaptic      |          |
|                    |                    | response      |          |
|                    |                    | after         |          |
|                    |                    | receiving a   |          |
|                    |                    | spike, we     |          |
|                    |                    | want to have  |          |
|                    |                    | a box         |          |
|                    |                    | response,     |          |
|                    |                    | i.e. a        |          |
|                    |                    | constant post |          |
|                    |                    | synaptic      |          |
|                    |                    | current for a |          |
|                    |                    | specified     |          |
|                    |                    | amount of     |          |
|                    |                    | time.         |          |
+--------------------+--------------------+---------------+----------+
| tutorial_21_dvs_in | DVS Camera         | This tutorial | Sensor   |
| putfile            | integration        | provides a    |          |
|                    |                    | basic example |          |
|                    |                    | of building   |          |
|                    |                    | an input      |          |
|                    |                    | layer to      |          |
|                    |                    | accept DVS    |          |
|                    |                    | Input, and    |          |
|                    |                    | then reading  |          |
|                    |                    | a DVS file    |          |
|                    |                    | and sending   |          |
|                    |                    | spikes into   |          |
|                    |                    | the input     |          |
|                    |                    | layer.        |          |
|                    |                    | Additionally  |          |
|                    |                    | it shows how  |          |
|                    |                    | to probe the  |          |
|                    |                    | activity of   |          |
|                    |                    | that input    |          |
|                    |                    | layer and     |          |
|                    |                    | recreate the  |          |
|                    |                    | DVS input     |          |
|                    |                    | (plot)        |          |
+--------------------+--------------------+---------------+----------+

.. _nxcore-8:

NxCore
''''''

+--------------------+--------------------+---------------+----------+
| Python Tutorials   | Feature Covered    | Description   | Category |
+====================+====================+===============+==========+
| tutorial_04b_synap | Synaptic Delay,    | This tutorial | SNIP     |
| tic_delays_spike_i | Spike Injection,   | introduces    |          |
| njection_snip      | SNIPs              | synaptic      |          |
|                    |                    | delays and    |          |
|                    |                    | will          |          |
|                    |                    | demonstrate   |          |
|                    |                    | how they work |          |
|                    |                    | and how to    |          |
|                    |                    | configure     |          |
|                    |                    | them using    |          |
|                    |                    | SNIPs         |          |
+--------------------+--------------------+---------------+----------+
| tutorial_19b_soma_ | Soma Activity,     | This tutorial | SNIP     |
| traces_spike_injec | Spike Injection,   | introduces    |          |
| tion_snip          | SNIPs              | soma traces   |          |
|                    |                    | and will      |          |
|                    |                    | demonstrate   |          |
|                    |                    | how they work |          |
|                    |                    | and how to    |          |
|                    |                    | configure     |          |
|                    |                    | them using    |          |
|                    |                    | SNIPs         |          |
+--------------------+--------------------+---------------+----------+

.. _major-bug-fixes-for-release-8:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Learning enabled connections could now be placed along with
   non-learning connections during creation of a NxNet network

.. _any-newsupported-platformshardware-1:

Any new/supported platforms/Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  NxSDK now supports the Kapoho Bay USB Platform and users can run
   NxNet and NxCore tutorials over this pluggable USB device
-  NxSDK now supports networks that span multiple Loihi chips
-  For those that were running in non-SLURM mode, here is the new
   procedure:

   -  run export **NXSDKHOST=<IP_OR_HOSTNAME_OF_BOARD>** and execute
      your scripts without setting **SLURM=1**

.. _api-changesdeprecations-6:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  To be compatible with future versions of python, we have updated the
   variable name async>aSync and all uses thereto
-  python_nx is deprecated. NxSDK needs Python 3.5.2 as some modules are
   compiled against this version.
-  External Git access is deprecated. Please use the pip installation
   procedure mentioned in the setup to install and use NxSDK 0.7
   release. We will introduce a new way to contribute code to NxSDK
   shortly.

.. _section-13:

0.5.5
-----

--------------

Installation
~~~~~~~~~~~~

Please ensure you have access to YOURVM.research.intel-research.net.
Packaged release resides in **/nfs/ncl/git/NxSDK.git**

1. Run the following to clone the repository:

::

   mkdir nxsdk
   git clone /nfs/ncl/git/NxSDK.git nxsdk 
   cd nxsdk
   # You might optionally add the git remote URL: ssh://user@YOURVM.research.intel-research.net:/nfs/ncl/git/NxSDK.git

2. All host and embedded binaries and libraries are pre-built. Set
   **PROJECT_ITOOLS** environment variable to point to .itools file in
   the git repo: ``export PROJECT_ITOOLS=$HOME/nxsdk/.itools``

3. Setup your python virtual environment:
   ``python_nx -m venv nxsdk_env``

4. Activate your python virtual environment:
   ``source nxsdk_env/bin/activate``

5. Install the python dependencies by running:
   ``sudo pip3 install -r requirements.txt``

Getting Started
~~~~~~~~~~~~~~~

To run a **NxNet** tutorial, execute:

::

   SLURM=1 python_nx -m examples.tutorials.nxnet.tutorial_01_single_compartment_with_bias

All NxNet tutorials are under **nxsdk/examples/tutorials/nxnet**

To run a **NxCore** tutorial, execute

::

   SLURM=1 python_nx -m examples.tutorials.nxcore.tutorial_01_single_compartment_with_bias

All NxCore tutorials are under **nxsdk/examples/tutorials/nxcore**

[Optional] If you are not using SLURM, execute on a terminal:

::

   srun /nfs/ncl/tools/bin/n2driverservice | tee -i n2driver.log 
   # Then you can submit your python job as: 
   python_nx -m examples.tutorials.nxnet.tutorial_01_single_compartment_with_bias

.. _to-view-documentation-1:

To View Documentation:
~~~~~~~~~~~~~~~~~~~~~~

::

   cd docs
   # You can start a simple http server or serve the docs via web-server like Apache or Nginx
   python3 -m http.server

Release Notes 0.5.5
~~~~~~~~~~~~~~~~~~~

.. _new-featuresimprovements-12:

New Features/Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _nxnet-9:

NxNet
'''''

-  Added support for DVS Camera using dvsSpikeGen interface
-  Added caching to speedup processing of time-series data from synapse
   probes (for weight, delay, tag).
-  Changed name of ‘weights’, ‘delays’, ‘tags’ in
   NxNet.createConnectionGroup to ‘weight’, ‘delay’, ‘tag’

.. _section-14:

0.5.4
-----

--------------

.. _new-featuresimprovements-13:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-10:

NxNet
^^^^^

-  Improved time consumed during synapse compilation for networks with
   large number of synapses
-  Improved time spent during sending spike inputs while using
   ``SpikeGenProcess.addSpikes`` (Preparing Input phase)
-  Added basic support for multiple connections with different learning
   rules going to the same destination compartment.
-  Network compilation time improvements.

.. _nxcore-9:

NxCore
^^^^^^

-  Improved time spent during sending spike inputs while using
   ``BasicSpikeGenerator.addSpike`` (Preparing Input phase)

.. _major-bug-fixes-for-release-9:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Removed compiler restriction that if connections exist in the
   network, then all cores with compartments must have connections.
-  Fixed the problem of allocating an unnecessarily large number of
   input axons causing the synapseMap to fill up.

.. _section-15:

0.5.3
-----

--------------

.. _new-featuresimprovements-14:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-11:

NxNet
^^^^^

-  Add support for no-factor learning rules (e.g. ``dw = -2*u0``)
-  Default numTagBits and numDelayBits to 0
-  Support multiple literals in an equation like ``dw = -2^3*x0*3*2``.
   There is also a new option: combineLiterals=True which will collapse
   all subsequent literals into one (the old behavior). For now, we can
   keep them separate. This will be necessary sometimes.
-  MicroCode class now determines the stdpProfileCfg.usesXEpoch
   parameter automatically.
-  Support x0, y0, r0 as explicit binary factors (before it was only
   used as dependency factors to gate the evaluation of a product).
-  Process (Channel) based spike injection is turned on by default

Known issues/limitations
~~~~~~~~~~~~~~~~~~~~~~~~

-  **NxNet**:

   -  Multiple connections with different learning rules going to the
      same destination compartment are not supported. Error will be
      raised.
   -  There is a limit to the number of synapses connected to a single
      compartment. The limit varies depending on the network
      configuration. Error will be raised.
   -  Learning connections must be placed at the end (i.e. after
      non-learning connections) explicitly by the user in NxNet.
      Currently no error is raised by compiler.
   -  Learning rules must have a pre-trace component. Learning rules
      without any pre-trace term will not learn. Workaround is to
      include the dummy term ``0*x1*u0``.
   -  Reward learning is not supported.

.. _major-bug-fixes-for-release-10:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  NxNet API documentation is fixed
-  Fixed a validation bug in which MicroCode should have failed if the
   user specifies multiple decimal exponents per learning rule

.. _section-16:

0.5.2
-----

--------------

.. _new-featuresimprovements-15:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-12:

NxNet
^^^^^

-  LCA documentation

+-------------+-----------------------+------------------+------------+
| New         | Feature Covered       | Description      | Category   |
| Tutorials   |                       |                  |            |
+=============+=======================+==================+============+
| 18b         | Learning on           | This tutorial    | Learning   |
|             | connections from      | illustrates the  |            |
|             | spike generator to    | use of the ESTDP |            |
|             | compartment           | learning rule.   |            |
|             |                       | The difference   |            |
|             |                       | between tutorial |            |
|             |                       | 18 and 18b is    |            |
|             |                       | that 18b has the |            |
|             |                       | spike generators |            |
|             |                       | directly         |            |
|             |                       | connected to     |            |
|             |                       | compartment 3    |            |
|             |                       | and learning     |            |
|             |                       | occurs on the    |            |
|             |                       | connection       |            |
|             |                       | between a spike  |            |
|             |                       | generator and    |            |
|             |                       | c3.              |            |
+-------------+-----------------------+------------------+------------+

.. _known-issueslimitations-1:

Known issues/limitations
~~~~~~~~~~~~~~~~~~~~~~~~

-  **NxNet**:

   -  Multiple connections with different learning rules going to the
      same destination compartment are not supported. Error will be
      raised.
   -  There is a limit to the number of synapses connected to a single
      compartment. The limit varies depending on the network
      configuration. Error will be raised.

.. _major-bug-fixes-for-release-11:

Major bug fixes for release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  NxNet tutorials now work properly without a GUI
-  NxNet tutorial 04 gave an error due to improper use of the basic
   spike generator. The tutorial has been updated to use the NxNet spike
   generator.

.. _section-17:

0.5.1
-----

--------------

.. _new-featuresimprovements-16:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-13:

NxNet
^^^^^

-  Spike injection is now supported using createSpikeGenProcess,
   createSpikeInputPort and createSpikeInputPortGroup APIs
-  Convenience methods to set compartmentCurrentTimeConstant,
   compartmentVoltageTimeConstant, compartmentThreshold and bias are
   supported in Compartment and CompartmentPrototypes
-  NxNet now supports the run and disconnect api directly. Users do not
   have to instantiate N2Compiler or board to run a network
-  Importing all necessary classes to create and run a network could now
   be done via “import nxsdk.arch.n2a as nx”. See example nxnet
   tutorials
-  Preliminary support for multi-compartment neurons has been added

+-------------+-----------------------+------------------+------------+
| New         | Feature Covered       | Description      | Category   |
| Tutorials   |                       |                  |            |
+=============+=======================+==================+============+
| 19          | Demonstrate **soma    | In this tutorial | Compartmen |
|             | traces**              | we configure a   | ts         |
|             | configuration         | compartmentGroup |            |
|             |                       | with 6           |            |
|             |                       | compartments     |            |
|             |                       | driven by input  |            |
|             |                       | spikes and       |            |
|             |                       | activated        |            |
|             |                       | homeostasis. We  |            |
|             |                       | introduce soma   |            |
|             |                       | traces and will  |            |
|             |                       | demonstrate how  |            |
|             |                       | they work and    |            |
|             |                       | how to configure |            |
|             |                       | them. The trace  |            |
|             |                       | of the current,  |            |
|             |                       | spikes and soma  |            |
|             |                       | for different    |            |
|             |                       | time constants   |            |
|             |                       | is probed        |            |
+-------------+-----------------------+------------------+------------+
| 20          | Demonstrate           | This tutorial    | Compartmen |
|             | **homeostasis**       | introduces the   | ts         |
|             | feature of the        | homeostasis      |            |
|             | **soma** threshold    | feature of the   |            |
|             |                       | soma threshold   |            |
|             |                       | and will         |            |
|             |                       | demonstrate how  |            |
|             |                       | it can be        |            |
|             |                       | configured and   |            |
|             |                       | how it works.    |            |
|             |                       | Different        |            |
|             |                       | examples show    |            |
|             |                       | the possible     |            |
|             |                       | solutions to     |            |
|             |                       | change the       |            |
|             |                       | membrane         |            |
|             |                       | threshold of     |            |
|             |                       | compartments     |            |
+-------------+-----------------------+------------------+------------+

.. _nxcore-10:

NxCore
^^^^^^

-  SNIP support has been added (back)
-  API to convert chip number to physical chip ID to call within SNIP
   code ``ChipId nx_nth_chipid(uint16_t chip_number);``

+------------------+----------------------+----------------+-----------+
| New Tutorials    | Feature Covered      | Description    | Category  |
+==================+======================+================+===========+
| 04b              | Creating a 1 -> 4    | This tutorial  | SNIP,     |
|                  | **synaptic fanout**  | shows a number | Connectiv |
|                  | driven by a **SNIP   | of useful      | ity       |
|                  | based spike          | components:    | and Flow  |
|                  | generator**.         | setting up and |           |
|                  | Configuring each     | using a SNIP   |           |
|                  | **synapse** with a   | based spike    |           |
|                  | different delay      | generator,     |           |
|                  | while ellucidating   | configuring    |           |
|                  | the relationship     | synaptic       |           |
|                  | between **dendritic  | fanout with    |           |
|                  | compartments** and   | delay and      |           |
|                  | **delay precision**  | understanding  |           |
|                  |                      | the            |           |
|                  |                      | hardware-based |           |
|                  |                      | relationships  |           |
|                  |                      | between        |           |
|                  |                      | dendritic      |           |
|                  |                      | compartments   |           |
|                  |                      | and delay      |           |
|                  |                      | precision      |           |
+------------------+----------------------+----------------+-----------+
| 19b              | Demonstrate **soma   | In this        | SNIP,     |
|                  | traces**             | tutorial we    | Compartme |
|                  | configuration with   | configure 6    | nts       |
|                  | **SNIP based spike   | cores with 1   |           |
|                  | injection** and      | compartments   |           |
|                  | **channels**         | driven by      |           |
|                  |                      | input spikes   |           |
|                  |                      | and activated  |           |
|                  |                      | homeostasis.   |           |
|                  |                      | We introduce   |           |
|                  |                      | soma traces    |           |
|                  |                      | and will       |           |
|                  |                      | demonstrate    |           |
|                  |                      | how they work  |           |
|                  |                      | and how to     |           |
|                  |                      | configure      |           |
|                  |                      | them. The      |           |
|                  |                      | trace of the   |           |
|                  |                      | current,       |           |
|                  |                      | spikes and     |           |
|                  |                      | soma for       |           |
|                  |                      | different time |           |
|                  |                      | constants is   |           |
|                  |                      | probed         |           |
+------------------+----------------------+----------------+-----------+

.. _known-issues-6:

Known Issues
~~~~~~~~~~~~

-  **NxNet**:

   -  Multiple connections with different learning rules going to the
      same destination compartment are not supported. Error will be
      raised.
   -  There is a limit to the number of synapses connected to a single
      compartment. The limit varies depending on the network
      configuration. Error will be raised.

Major Bug Fixes
~~~~~~~~~~~~~~~

-  Fixed bugs related to unreliable SNIP communication over channels

.. _section-18:

0.5
---

--------------

.. _new-featuresimprovements-17:

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _nxnet-14:

NxNet
^^^^^

-  0.5 introduces the NxNet API which provides a higher level of
   abstraction over the NxCore API for network configuration:

   -  The user is no longer required to configure neural networks in
      terms of low level registers but in terms of high level
      compartments and connections.
   -  Compartment and CompartmentGroups can be configured using
      CompartmentPrototypes.
   -  Connections and ConnectionGroups can be configured using
      ConnectionPrototypes.
   -  Synaptic learning rules including dynamic state equations and
      pre-, post- trace configuration can be configured and assigned to
      connections.
   -  Probes to monitor the evolution of network state (compartment
      voltage, current, … synaptic weights, delays, …).

-  Underlying hardware limit validations are enforced during network
   compilation and user gets informed in case of violations
-  Tutorials were added to show feature availability and parity with
   NxCore

+-------------+-----------------------+------------------+------------+
| Tutorials   | Feature Covered       | Description      | Category   |
+=============+=======================+==================+============+
| 01          | Creating a **single   | Within this      | Compartmen |
|             | compartment neuron**  | tutorial, you    | ts         |
|             | with a **bias**       | will learn how   |            |
|             | current               | to create a      |            |
|             |                       | single           |            |
|             |                       | compartment      |            |
|             |                       | neuron,          |            |
|             |                       | configure the    |            |
|             |                       | bias current and |            |
|             |                       | monitor the      |            |
|             |                       | resulting spike  |            |
|             |                       | behavior.        |            |
+-------------+-----------------------+------------------+------------+
| 02          | Creating a **single   | The second       | Connectivi |
|             | compartment neuron**  | tutorial builds  | ty         |
|             | and **connect** it to | on the first,    | and Flow   |
|             | two other             | showing you now  |            |
|             | compartments in the   | how to connect   |            |
|             | same neurocore        | compartments     |            |
|             |                       | using connection |            |
|             |                       | prototypes and   |            |
|             |                       | connections      |            |
+-------------+-----------------------+------------------+------------+
| 03          | Creating a **single   | The third        | Connectivi |
|             | compartment neuron**  | tutorial builds  | ty         |
|             | and **connect** it to | on the first two | and Flow   |
|             | three other           | to demonstrate   |            |
|             | compartments in       | how to place     |            |
|             | **different logical   | compartments on  |            |
|             | neurocores**          | different        |            |
|             |                       | neurocores and   |            |
|             |                       | connect them to  |            |
|             |                       | create the       |            |
|             |                       | desired network  |            |
+-------------+-----------------------+------------------+------------+
| 04          | Creating a 1 -> 4     | This tutorial    | Connectivi |
|             | **connection          | shows a number   | ty         |
|             | fanout**, driven by a | of useful        | and Flow   |
|             | **spike generator**.  | components:      |            |
|             | Configuring each      | setting up and   |            |
|             | **connection** with a | using a spike    |            |
|             | different delay while | generator,       |            |
|             | elucidating the       | configuring      |            |
|             | relationship between  | synaptic fanout  |            |
|             | **dendritic           | with delay and   |            |
|             | compartments** and    | understanding    |            |
|             | **delay precision**   | the              |            |
|             |                       | hardware-based   |            |
|             |                       | relationships    |            |
|             |                       | between          |            |
|             |                       | dendritic        |            |
|             |                       | compartments and |            |
|             |                       | delay precision  |            |
+-------------+-----------------------+------------------+------------+
| 08          | Demonstrates **static | Configures four  | Compartmen |
|             | and stochastic        | different        | ts         |
|             | refractory delays**   | single-compartme |            |
|             |                       | nt               |            |
|             |                       | neurons with     |            |
|             |                       | different        |            |
|             |                       | profiles. Each   |            |
|             |                       | profile utilizes |            |
|             |                       | a different      |            |
|             |                       | refractory       |            |
|             |                       | delay. The       |            |
|             |                       | tutorial then    |            |
|             |                       | sets up probes   |            |
|             |                       | to monitor the   |            |
|             |                       | results          |            |
+-------------+-----------------------+------------------+------------+
| 12          | Demonstrates how to   | Utilizing 4      | Connectivi |
|             | configure **synaptic  | different        | ty         |
|             | weight precision**    | connections,     | and Flow   |
|             |                       | this tutorial    |            |
|             |                       | shows how        |            |
|             |                       | weights can be   |            |
|             |                       | configured to    |            |
|             |                       | achieve          |            |
|             |                       | different        |            |
|             |                       | results          |            |
+-------------+-----------------------+------------------+------------+
| 14          | Demonstrate how to    | Within this      | Compartmen |
|             | **inject noise** into | tutorial, we     | ts         |
|             | compartments and      | inject noise     |            |
|             | **randomize current** | into current     |            |
|             | and/or **voltage**    | variable         |            |
|             |                       | (randomizeCurren |            |
|             |                       | t)               |            |
|             |                       | for a            |            |
|             |                       | compartment in   |            |
|             |                       | logical core 0   |            |
|             |                       | and into voltage |            |
|             |                       | variable         |            |
|             |                       | (randomizeVoltag |            |
|             |                       | e)               |            |
|             |                       | for a            |            |
|             |                       | compartment in   |            |
|             |                       | logical core 1   |            |
|             |                       | and probe the    |            |
|             |                       | states to show   |            |
|             |                       | impact of noise  |            |
|             |                       | injection        |            |
+-------------+-----------------------+------------------+------------+
| 18          | Demonstrate **ESTDP   | This tutorial    | Learning   |
|             | learning** on a       | illustrates the  |            |
|             | learning enabled      | use of the ESTDP |            |
|             | connection between    | learning rule.   |            |
|             | two compartments      | Two connections  |            |
|             |                       | are connected    |            |
|             |                       | via independent  |            |
|             |                       | synapses to the  |            |
|             |                       | same compartment |            |
|             |                       | 3, one being a   |            |
|             |                       | learning enabled |            |
|             |                       | connection. We   |            |
|             |                       | configure        |            |
|             |                       | pre-synaptic     |            |
|             |                       | traces (X) which |            |
|             |                       | act as inputs    |            |
|             |                       | into learning    |            |
|             |                       | rule and         |            |
|             |                       | post-synaptic    |            |
|             |                       | traces (Y).      |            |
|             |                       | Connection       |            |
|             |                       | weights are      |            |
|             |                       | probed to show   |            |
|             |                       | learning         |            |
|             |                       | behaviour        |            |
+-------------+-----------------------+------------------+------------+

.. _nxcore-11:

NxCore
^^^^^^

-  Performance of Spike generation was improved using message passing
-  Energy probe was added

+------------------+----------------------+----------------+-----------+
| New Tutorials    | Feature Covered      | Description    | Category  |
+==================+======================+================+===========+
| 03               | Creating a **single  | The third      | Connectiv |
|                  | compartment neuron** | tutorial       | ity       |
|                  | and **connect** it   | builds on the  | and Flow  |
|                  | to three other       | first two to   |           |
|                  | compartments in      | demonstrate    |           |
|                  | **different          | how to place   |           |
|                  | neurocores**         | compartments   |           |
|                  |                      | on different   |           |
|                  |                      | neurocores and |           |
|                  |                      | connect them   |           |
|                  |                      | to create the  |           |
|                  |                      | desired        |           |
|                  |                      | network        |           |
+------------------+----------------------+----------------+-----------+
| 13               | Demonstrates         | Within this    | Compartme |
|                  | **axonal delay** on  | tutorial, we   | nts       |
|                  | two different        | set different  |           |
|                  | compartments         | axonal delays  |           |
|                  |                      | for each       |           |
|                  |                      | neuron to      |           |
|                  |                      | illustrate the |           |
|                  |                      | delay from     |           |
|                  |                      | when the       |           |
|                  |                      | neuron         |           |
|                  |                      | generates the  |           |
|                  |                      | spike to when  |           |
|                  |                      | the spike is   |           |
|                  |                      | actually sent  |           |
|                  |                      | out to         |           |
|                  |                      | connected      |           |
|                  |                      | compartments   |           |
+------------------+----------------------+----------------+-----------+
| 14               | Demonstrate how to   | Within this    | Compartme |
|                  | **inject noise**     | tutorial, we   | nts       |
|                  | into compartments    | inject noise   |           |
|                  | and **randomize      | into current   |           |
|                  | current** and/or     | variable (u)   |           |
|                  | **voltage**          | for            |           |
|                  |                      | compartment 0  |           |
|                  |                      | of core 0 and  |           |
|                  |                      | into voltage   |           |
|                  |                      | variable (v)   |           |
|                  |                      | for            |           |
|                  |                      | compartment 0  |           |
|                  |                      | of core 1 and  |           |
|                  |                      | probe the      |           |
|                  |                      | states to show |           |
|                  |                      | impact of      |           |
|                  |                      | noise          |           |
|                  |                      | injection      |           |
+------------------+----------------------+----------------+-----------+
| 15               | Demonstrate how to   | In this        | Compartme |
|                  | create a             | tutorial we    | nts       |
|                  | **multi-compartment  | will create a  |           |
|                  | neuron** and         | two            |           |
|                  | demonstrate basic    | compartment    |           |
|                  | principle how        | neuron by      |           |
|                  | information flows    | configuring    |           |
|                  | from one compartment | stackOut and   |           |
|                  | via **stack** to     | stackInt and   |           |
|                  | another one          | using basic    |           |
|                  |                      | ADD join       |           |
|                  |                      | function       |           |
+------------------+----------------------+----------------+-----------+
| 16               | Demonstrate the      | In this        | Compartme |
|                  | impact of the        | tutorial we    | nts       |
|                  | **tepoch setting**   | configure two  |           |
|                  | when                 | independent    |           |
|                  | **axonal_delay** is  | compartments   |           |
|                  | enabled              | with same      |           |
|                  |                      | axonal delay   |           |
|                  |                      | each connected |           |
|                  |                      | to a separate  |           |
|                  |                      | compartment    |           |
|                  |                      | through a      |           |
|                  |                      | synapse. Using |           |
|                  |                      | different v    |           |
|                  |                      | threshold      |           |
|                  |                      | values, we     |           |
|                  |                      | demonstrate    |           |
|                  |                      | the impact of  |           |
|                  |                      | tepoch setting |           |
|                  |                      | on when spikes |           |
|                  |                      | occur within   |           |
|                  |                      | the tepoch     |           |
+------------------+----------------------+----------------+-----------+
| 17               | Demonstrate **spike  | In this        | Compartme |
|                  | probes** to capture  | tutorial we    | nts       |
|                  | the occurrence and   | configure the  |           |
|                  | count of **spikes**  | spike probe on |           |
|                  | by configuring       | a compartment  |           |
|                  | different            | and configure  |           |
|                  | **SpikeProbeConditio | three          |           |
|                  | ns**                 | different      |           |
|                  |                      | spike probe    |           |
|                  |                      | conditions to  |           |
|                  |                      | capture the    |           |
|                  |                      | occurrence and |           |
|                  |                      | count of       |           |
|                  |                      | spikes over    |           |
|                  |                      | specified      |           |
|                  |                      | intervals of   |           |
|                  |                      | time           |           |
+------------------+----------------------+----------------+-----------+
| 18               | Demonstrate          | This tutorial  | Learning  |
|                  | **learning** to      | illustrates    |           |
|                  | drive change in      | the use of     |           |
|                  | synaptic weights     | learning rules |           |
|                  | using **ESTDP        | using two      |           |
|                  | rules**              | input axons,   |           |
|                  |                      | one with       |           |
|                  |                      | learning       |           |
|                  |                      | enabled, are   |           |
|                  |                      | connected via  |           |
|                  |                      | independent    |           |
|                  |                      | synapses to    |           |
|                  |                      | same           |           |
|                  |                      | compartment.   |           |
|                  |                      | Various        |           |
|                  |                      | learning rules |           |
|                  |                      | are configured |           |
|                  |                      | along with     |           |
|                  |                      | various spike  |           |
|                  |                      | timings to     |           |
|                  |                      | illustrate the |           |
|                  |                      | affects of     |           |
|                  |                      | learning       |           |
+------------------+----------------------+----------------+-----------+
| 18b              | Demonstrate **delay  | This tutorial  | Learning  |
|                  | learning** to drive  | illustrates    |           |
|                  | change in synaptic   | the use of     |           |
|                  | weights using        | learning rules |           |
|                  | **ESTDP rules**      | using two      |           |
|                  |                      | input axons,   |           |
|                  |                      | one with       |           |
|                  |                      | learning       |           |
|                  |                      | enabled, are   |           |
|                  |                      | connected via  |           |
|                  |                      | independent    |           |
|                  |                      | synapses to    |           |
|                  |                      | same           |           |
|                  |                      | compartment.   |           |
|                  |                      | Various        |           |
|                  |                      | learning rules |           |
|                  |                      | are configured |           |
|                  |                      | along with     |           |
|                  |                      | various spike  |           |
|                  |                      | timings to     |           |
|                  |                      | illustrate the |           |
|                  |                      | affects of     |           |
|                  |                      | learning.      |           |
|                  |                      | Additionally,  |           |
|                  |                      | synaptic delay |           |
|                  |                      | is added to    |           |
|                  |                      | demonstrate    |           |
|                  |                      | the delay in   |           |
|                  |                      | learning       |           |
|                  |                      | behavior       |           |
+------------------+----------------------+----------------+-----------+
| 19               | Demonstrate **soma   | In this        | Compartme |
|                  | traces**             | tutorial we    | nts       |
|                  | configuration        | configure 6    |           |
|                  |                      | cores with 1   |           |
|                  |                      | compartments   |           |
|                  |                      | driven by      |           |
|                  |                      | input spikes   |           |
|                  |                      | and activated  |           |
|                  |                      | homeostasis.   |           |
|                  |                      | We introduce   |           |
|                  |                      | soma traces    |           |
|                  |                      | and will       |           |
|                  |                      | demonstrate    |           |
|                  |                      | how they work  |           |
|                  |                      | and how to     |           |
|                  |                      | configure      |           |
|                  |                      | them. The      |           |
|                  |                      | trace of the   |           |
|                  |                      | current,       |           |
|                  |                      | spikes and     |           |
|                  |                      | soma for       |           |
|                  |                      | different time |           |
|                  |                      | constants is   |           |
|                  |                      | probed         |           |
+------------------+----------------------+----------------+-----------+
| 20               | Demonstrate          | This tutorial  | Compartme |
|                  | **homeostasis**      | introduces the | nts       |
|                  | feature of the       | homeostasis    |           |
|                  | **soma threshold**   | feature of the |           |
|                  |                      | soma threshold |           |
|                  |                      | and will       |           |
|                  |                      | demonstrate    |           |
|                  |                      | how it can be  |           |
|                  |                      | configured and |           |
|                  |                      | how it works.  |           |
|                  |                      | Different      |           |
|                  |                      | examples show  |           |
|                  |                      | the possible   |           |
|                  |                      | solutions to   |           |
|                  |                      | change the     |           |
|                  |                      | membrane       |           |
|                  |                      | threshold of   |           |
|                  |                      | compartments   |           |
+------------------+----------------------+----------------+-----------+

.. _known-issues-7:

Known Issues
~~~~~~~~~~~~

-  **NxCore**:

   -  Using Spike probes on compartments along with manually configuring
      axons is unsupported. Use createDiscreteAxon API from AxonCompiler
      instead.
   -  SNIP communication over channels might give unexpected results if
      there are outstanding reads to be serviced.
   -  bapSrc being set to 0 or 1 appears to make no difference when
      bapAction is set to 2 (Propagate backwards).

-  **NxNet**:

   -  Core-agnostic programming is not yet fully supported. The user is
      still required to assign logicalCoreIds to high level compartments
      in the network but the compiler will automatically determine all
      associated register entry placements.
   -  Synaptic encoding/compression:

      -  The user has the option to select a synaptic compression mode
         (sparse, runlength, dense) manually through the NxNet API or
         let NxCompiler chose a compression mode automatically.
      -  Currently NxCompiler always selects ‘sparse’ as the default
         synaptic compression mode.

   -  All connections/connection-groups created from ConnectionPrototype
      with learning-enabled should be defined after regular
      connections/connection-groups (which do not have learning
      enabled).
   -  Mapping tables from high level NxNet nodeIds to register addresses
      (aka resource map) not yet available. Thus network state cannot be
      interactively modified by SNIPs when using high level API.

-  **General**:

   -  Chip to chip communication is currently not supported.
   -  Mesh deadlocks during high levels of spike activity between
      neurons. Typically throws “ERROR: nx_wait_barrier timeout” error
      message. Resolution: Use avoid-deadlock=1 in compiler options to
      enable a workaround (this will potentially slow down the
      execution).

.. _major-bug-fixes-1:

Major Bug Fixes
~~~~~~~~~~~~~~~

-  Mesh deadlocks during high levels of spike activity between neurons.
   Typically throws “ERROR: nx_wait_barrier timeout” error message.
   Resolution: Use avoid-deadlock=1 in compiler options to enable a
   workaround (this will potentially slow down the execution).
-  Incorrect weight scaling by wgtExp for negative wgtExp values.

New supported platforms/Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Nahuku** boards will be supported for running NxNet and NxCore
   applications.

.. _api-changesdeprecations-7:

API Changes/Deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

-  0.5 introduces **NxNet** API which provides a higher level of
   abstraction to create and run a spiking neural network in Python.
   NxCore API is still available for neuro-core and board level usage,
   however using NxNet to model networks and experiments is **highly
   encouraged**.
