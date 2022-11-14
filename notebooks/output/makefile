PROGRAM = main

SRCS = brianlib/randomkit/randomkit.c code_objects/L1_spike_resetter_codeobject.cpp code_objects/L1_spike_thresholder_codeobject.cpp code_objects/L1_stateupdater_codeobject.cpp code_objects/L2_spike_resetter_codeobject.cpp code_objects/L2_spike_thresholder_codeobject.cpp code_objects/L2_stateupdater_codeobject.cpp code_objects/L3_increase_threshold_resetter_codeobject.cpp code_objects/L3_increase_threshold_thresholder_codeobject.cpp code_objects/L3_spike_resetter_codeobject.cpp code_objects/L3_spike_thresholder_codeobject.cpp code_objects/L3_stateupdater_codeobject.cpp code_objects/SpikeMonitor_L1_codeobject.cpp code_objects/SpikeMonitor_L2_codeobject.cpp code_objects/SpikeMonitor_L3_codeobject.cpp code_objects/StateMonitor_L1_codeobject.cpp code_objects/StateMonitor_L2_codeobject.cpp code_objects/StateMonitor_L3_codeobject.cpp code_objects/StateMonitor_Syn23_codeobject.cpp code_objects/Syn12_pre_codeobject.cpp code_objects/Syn12_pre_push_spikes.cpp code_objects/Syn12_summed_variable_sum_w_post_codeobject.cpp code_objects/Syn12_synapses_create_array_codeobject.cpp code_objects/Syn23_post_codeobject.cpp code_objects/Syn23_post_push_spikes.cpp code_objects/Syn23_pre_codeobject.cpp code_objects/Syn23_pre_push_spikes.cpp code_objects/Syn23_stateupdater_codeobject.cpp code_objects/Syn23_summed_variable_sum_w_post_codeobject.cpp code_objects/Syn23_synapses_create_generator_codeobject.cpp code_objects/Syn33_pre_codeobject.cpp code_objects/Syn33_pre_push_spikes.cpp code_objects/Syn33_synapses_create_generator_codeobject.cpp code_objects/after_run_L1_spike_thresholder_codeobject.cpp code_objects/after_run_L2_spike_thresholder_codeobject.cpp code_objects/after_run_L3_increase_threshold_thresholder_codeobject.cpp code_objects/after_run_L3_spike_thresholder_codeobject.cpp code_objects/before_run_Syn12_pre_push_spikes.cpp code_objects/before_run_Syn23_post_push_spikes.cpp code_objects/before_run_Syn23_pre_push_spikes.cpp code_objects/before_run_Syn33_pre_push_spikes.cpp main.cpp network.cpp objects.cpp run.cpp synapses_classes.cpp
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/dynamic_array.h code_objects/L1_spike_resetter_codeobject.h code_objects/L1_spike_thresholder_codeobject.h code_objects/L1_stateupdater_codeobject.h code_objects/L2_spike_resetter_codeobject.h code_objects/L2_spike_thresholder_codeobject.h code_objects/L2_stateupdater_codeobject.h code_objects/L3_increase_threshold_resetter_codeobject.h code_objects/L3_increase_threshold_thresholder_codeobject.h code_objects/L3_spike_resetter_codeobject.h code_objects/L3_spike_thresholder_codeobject.h code_objects/L3_stateupdater_codeobject.h code_objects/SpikeMonitor_L1_codeobject.h code_objects/SpikeMonitor_L2_codeobject.h code_objects/SpikeMonitor_L3_codeobject.h code_objects/StateMonitor_L1_codeobject.h code_objects/StateMonitor_L2_codeobject.h code_objects/StateMonitor_L3_codeobject.h code_objects/StateMonitor_Syn23_codeobject.h code_objects/Syn12_pre_codeobject.h code_objects/Syn12_pre_push_spikes.h code_objects/Syn12_summed_variable_sum_w_post_codeobject.h code_objects/Syn12_synapses_create_array_codeobject.h code_objects/Syn23_post_codeobject.h code_objects/Syn23_post_push_spikes.h code_objects/Syn23_pre_codeobject.h code_objects/Syn23_pre_push_spikes.h code_objects/Syn23_stateupdater_codeobject.h code_objects/Syn23_summed_variable_sum_w_post_codeobject.h code_objects/Syn23_synapses_create_generator_codeobject.h code_objects/Syn33_pre_codeobject.h code_objects/Syn33_pre_push_spikes.h code_objects/Syn33_synapses_create_generator_codeobject.h code_objects/after_run_L1_spike_thresholder_codeobject.h code_objects/after_run_L2_spike_thresholder_codeobject.h code_objects/after_run_L3_increase_threshold_thresholder_codeobject.h code_objects/after_run_L3_spike_thresholder_codeobject.h code_objects/before_run_Syn12_pre_push_spikes.h code_objects/before_run_Syn23_post_push_spikes.h code_objects/before_run_Syn23_pre_push_spikes.h code_objects/before_run_Syn33_pre_push_spikes.h network.h objects.h run.h synapses_classes.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = -Ibrianlib/randomkit -I/home/alex/miniconda3/envs/mujoco/include -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/home/alex/miniconda3/envs/mujoco/lib -Wl,-R/home/alex/miniconda3/envs/mujoco/lib 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@