#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "randomkit.h"

#include "code_objects/L1_spike_resetter_codeobject.h"
#include "code_objects/L1_spike_thresholder_codeobject.h"
#include "code_objects/L1_stateupdater_codeobject.h"
#include "code_objects/L2_spike_resetter_codeobject.h"
#include "code_objects/L2_spike_thresholder_codeobject.h"
#include "code_objects/L2_stateupdater_codeobject.h"
#include "code_objects/L3_increase_threshold_resetter_codeobject.h"
#include "code_objects/L3_increase_threshold_thresholder_codeobject.h"
#include "code_objects/L3_spike_resetter_codeobject.h"
#include "code_objects/L3_spike_thresholder_codeobject.h"
#include "code_objects/L3_stateupdater_codeobject.h"
#include "code_objects/SpikeMonitor_L1_codeobject.h"
#include "code_objects/SpikeMonitor_L2_codeobject.h"
#include "code_objects/SpikeMonitor_L3_codeobject.h"
#include "code_objects/StateMonitor_L1_codeobject.h"
#include "code_objects/StateMonitor_L2_codeobject.h"
#include "code_objects/StateMonitor_L3_codeobject.h"
#include "code_objects/StateMonitor_Syn23_codeobject.h"
#include "code_objects/Syn12_pre_codeobject.h"
#include "code_objects/Syn12_pre_push_spikes.h"
#include "code_objects/Syn12_summed_variable_sum_w_post_codeobject.h"
#include "code_objects/Syn12_synapses_create_array_codeobject.h"
#include "code_objects/Syn23_post_codeobject.h"
#include "code_objects/Syn23_post_push_spikes.h"
#include "code_objects/Syn23_pre_codeobject.h"
#include "code_objects/Syn23_pre_push_spikes.h"
#include "code_objects/Syn23_stateupdater_codeobject.h"
#include "code_objects/Syn23_summed_variable_sum_w_post_codeobject.h"
#include "code_objects/Syn23_synapses_create_generator_codeobject.h"
#include "code_objects/Syn33_pre_codeobject.h"
#include "code_objects/Syn33_pre_push_spikes.h"
#include "code_objects/Syn33_synapses_create_generator_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    for (int i=0; i<1; i++)
	    rk_randomseed(brian::_mersenne_twister_states[i]);  // Note that this seed can be potentially replaced in main.cpp
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


