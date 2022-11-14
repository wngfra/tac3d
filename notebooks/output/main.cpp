#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/L1_spike_resetter_codeobject.h"
#include "code_objects/L1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_L1_spike_thresholder_codeobject.h"
#include "code_objects/L1_stateupdater_codeobject.h"
#include "code_objects/L2_spike_resetter_codeobject.h"
#include "code_objects/L2_spike_thresholder_codeobject.h"
#include "code_objects/after_run_L2_spike_thresholder_codeobject.h"
#include "code_objects/L2_stateupdater_codeobject.h"
#include "code_objects/L3_increase_threshold_resetter_codeobject.h"
#include "code_objects/L3_increase_threshold_thresholder_codeobject.h"
#include "code_objects/after_run_L3_increase_threshold_thresholder_codeobject.h"
#include "code_objects/L3_spike_resetter_codeobject.h"
#include "code_objects/L3_spike_thresholder_codeobject.h"
#include "code_objects/after_run_L3_spike_thresholder_codeobject.h"
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
#include "code_objects/before_run_Syn12_pre_push_spikes.h"
#include "code_objects/Syn12_summed_variable_sum_w_post_codeobject.h"
#include "code_objects/Syn12_synapses_create_array_codeobject.h"
#include "code_objects/Syn23_post_codeobject.h"
#include "code_objects/Syn23_post_push_spikes.h"
#include "code_objects/before_run_Syn23_post_push_spikes.h"
#include "code_objects/Syn23_pre_codeobject.h"
#include "code_objects/Syn23_pre_push_spikes.h"
#include "code_objects/before_run_Syn23_pre_push_spikes.h"
#include "code_objects/Syn23_stateupdater_codeobject.h"
#include "code_objects/Syn23_summed_variable_sum_w_post_codeobject.h"
#include "code_objects/Syn23_synapses_create_generator_codeobject.h"
#include "code_objects/Syn33_pre_codeobject.h"
#include "code_objects/Syn33_pre_push_spikes.h"
#include "code_objects/before_run_Syn33_pre_push_spikes.h"
#include "code_objects/Syn33_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>


        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0) 
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_L1_lastspike; i++)
                        {
                            _array_L1_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_L1_not_refractory; i++)
                        {
                            _array_L1_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_L2_lastspike; i++)
                        {
                            _array_L2_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_L2_not_refractory; i++)
                        {
                            _array_L2_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_L3_lastspike; i++)
                        {
                            _array_L3_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_L3_not_refractory; i++)
                        {
                            _array_L3_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_Syn12_sources; i++)
                        {
                            _array_Syn12_sources[i] = _static_array__array_Syn12_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_Syn12_targets; i++)
                        {
                            _array_Syn12_targets[i] = _static_array__array_Syn12_targets[i];
                        }
                        
        _run_Syn12_synapses_create_array_codeobject();
        _run_Syn23_synapses_create_generator_codeobject();
        _run_Syn33_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_num__array_StateMonitor_L1__indices; i++)
                        {
                            _array_StateMonitor_L1__indices[i] = _static_array__array_StateMonitor_L1__indices[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_StateMonitor_L2__indices; i++)
                        {
                            _array_StateMonitor_L2__indices[i] = _static_array__array_StateMonitor_L2__indices[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_StateMonitor_L3__indices; i++)
                        {
                            _array_StateMonitor_L3__indices[i] = _static_array__array_StateMonitor_L3__indices[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_StateMonitor_Syn23__indices; i++)
                        {
                            _array_StateMonitor_Syn23__indices[i] = _static_array__array_StateMonitor_Syn23__indices[i];
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_Syn12_pre_push_spikes();
        _before_run_Syn23_pre_push_spikes();
        _before_run_Syn33_pre_push_spikes();
        _before_run_Syn23_post_push_spikes();
        network.clear();
        network.add(&defaultclock, _run_StateMonitor_L1_codeobject);
        network.add(&defaultclock, _run_StateMonitor_L2_codeobject);
        network.add(&defaultclock, _run_StateMonitor_L3_codeobject);
        network.add(&defaultclock, _run_StateMonitor_Syn23_codeobject);
        network.add(&defaultclock, _run_Syn12_summed_variable_sum_w_post_codeobject);
        network.add(&defaultclock, _run_Syn23_summed_variable_sum_w_post_codeobject);
        network.add(&defaultclock, _run_L1_stateupdater_codeobject);
        network.add(&defaultclock, _run_L2_stateupdater_codeobject);
        network.add(&defaultclock, _run_L3_stateupdater_codeobject);
        network.add(&defaultclock, _run_Syn23_stateupdater_codeobject);
        network.add(&defaultclock, _run_L1_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_L2_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_L3_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_SpikeMonitor_L1_codeobject);
        network.add(&defaultclock, _run_SpikeMonitor_L2_codeobject);
        network.add(&defaultclock, _run_SpikeMonitor_L3_codeobject);
        network.add(&defaultclock, _run_L3_increase_threshold_thresholder_codeobject);
        network.add(&defaultclock, _run_Syn12_pre_push_spikes);
        network.add(&defaultclock, _run_Syn12_pre_codeobject);
        network.add(&defaultclock, _run_Syn23_pre_push_spikes);
        network.add(&defaultclock, _run_Syn23_pre_codeobject);
        network.add(&defaultclock, _run_Syn33_pre_push_spikes);
        network.add(&defaultclock, _run_Syn33_pre_codeobject);
        network.add(&defaultclock, _run_Syn23_post_push_spikes);
        network.add(&defaultclock, _run_Syn23_post_codeobject);
        network.add(&defaultclock, _run_L1_spike_resetter_codeobject);
        network.add(&defaultclock, _run_L2_spike_resetter_codeobject);
        network.add(&defaultclock, _run_L3_spike_resetter_codeobject);
        network.add(&defaultclock, _run_L3_increase_threshold_resetter_codeobject);
        network.run(1.502, report_progress, 10.0);
        _after_run_L1_spike_thresholder_codeobject();
        _after_run_L2_spike_thresholder_codeobject();
        _after_run_L3_spike_thresholder_codeobject();
        _after_run_L3_increase_threshold_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_SpikeMonitor_L1_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_SpikeMonitor_L2_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_SpikeMonitor_L3_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_Syn12_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_Syn23_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_Syn33_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_Syn23_post_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}