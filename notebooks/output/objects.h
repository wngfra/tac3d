
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network network;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_SpikeMonitor_L1_i;
extern std::vector<double> _dynamic_array_SpikeMonitor_L1_t;
extern std::vector<int32_t> _dynamic_array_SpikeMonitor_L2_i;
extern std::vector<double> _dynamic_array_SpikeMonitor_L2_t;
extern std::vector<int32_t> _dynamic_array_SpikeMonitor_L3_i;
extern std::vector<double> _dynamic_array_SpikeMonitor_L3_t;
extern std::vector<double> _dynamic_array_StateMonitor_L1_t;
extern std::vector<double> _dynamic_array_StateMonitor_L2_t;
extern std::vector<double> _dynamic_array_StateMonitor_L3_t;
extern std::vector<double> _dynamic_array_StateMonitor_Syn23_t;
extern std::vector<int32_t> _dynamic_array_Syn12__synaptic_post;
extern std::vector<int32_t> _dynamic_array_Syn12__synaptic_pre;
extern std::vector<double> _dynamic_array_Syn12_delay;
extern std::vector<int32_t> _dynamic_array_Syn12_N_incoming;
extern std::vector<int32_t> _dynamic_array_Syn12_N_outgoing;
extern std::vector<double> _dynamic_array_Syn12_w;
extern std::vector<int32_t> _dynamic_array_Syn23__synaptic_post;
extern std::vector<int32_t> _dynamic_array_Syn23__synaptic_pre;
extern std::vector<double> _dynamic_array_Syn23_c;
extern std::vector<double> _dynamic_array_Syn23_count;
extern std::vector<double> _dynamic_array_Syn23_delay;
extern std::vector<double> _dynamic_array_Syn23_delay_1;
extern std::vector<int32_t> _dynamic_array_Syn23_N_incoming;
extern std::vector<int32_t> _dynamic_array_Syn23_N_outgoing;
extern std::vector<double> _dynamic_array_Syn23_X;
extern std::vector<double> _dynamic_array_Syn23_X_condition;
extern std::vector<int32_t> _dynamic_array_Syn33__synaptic_post;
extern std::vector<int32_t> _dynamic_array_Syn33__synaptic_pre;
extern std::vector<double> _dynamic_array_Syn33_delay;
extern std::vector<int32_t> _dynamic_array_Syn33_N_incoming;
extern std::vector<int32_t> _dynamic_array_Syn33_N_outgoing;
extern std::vector<double> _dynamic_array_Syn33_w;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_L1__spikespace;
extern const int _num__array_L1__spikespace;
extern int32_t *_array_L1_i;
extern const int _num__array_L1_i;
extern double *_array_L1_lastspike;
extern const int _num__array_L1_lastspike;
extern char *_array_L1_not_refractory;
extern const int _num__array_L1_not_refractory;
extern double *_array_L1_v;
extern const int _num__array_L1_v;
extern int32_t *_array_L2__spikespace;
extern const int _num__array_L2__spikespace;
extern double *_array_L2_g_e;
extern const int _num__array_L2_g_e;
extern int32_t *_array_L2_i;
extern const int _num__array_L2_i;
extern double *_array_L2_lastspike;
extern const int _num__array_L2_lastspike;
extern char *_array_L2_not_refractory;
extern const int _num__array_L2_not_refractory;
extern double *_array_L2_sum_w;
extern const int _num__array_L2_sum_w;
extern double *_array_L2_v;
extern const int _num__array_L2_v;
extern int32_t *_array_L3__increase_thresholdspace;
extern const int _num__array_L3__increase_thresholdspace;
extern int32_t *_array_L3__spikespace;
extern const int _num__array_L3__spikespace;
extern double *_array_L3_g_e;
extern const int _num__array_L3_g_e;
extern double *_array_L3_g_i;
extern const int _num__array_L3_g_i;
extern int32_t *_array_L3_i;
extern const int _num__array_L3_i;
extern char *_array_L3_is_winner;
extern const int _num__array_L3_is_winner;
extern double *_array_L3_lastspike;
extern const int _num__array_L3_lastspike;
extern char *_array_L3_not_refractory;
extern const int _num__array_L3_not_refractory;
extern double *_array_L3_sum_w;
extern const int _num__array_L3_sum_w;
extern double *_array_L3_v;
extern const int _num__array_L3_v;
extern double *_array_L3_v_th;
extern const int _num__array_L3_v_th;
extern int32_t *_array_SpikeMonitor_L1__source_idx;
extern const int _num__array_SpikeMonitor_L1__source_idx;
extern int32_t *_array_SpikeMonitor_L1_count;
extern const int _num__array_SpikeMonitor_L1_count;
extern int32_t *_array_SpikeMonitor_L1_N;
extern const int _num__array_SpikeMonitor_L1_N;
extern int32_t *_array_SpikeMonitor_L2__source_idx;
extern const int _num__array_SpikeMonitor_L2__source_idx;
extern int32_t *_array_SpikeMonitor_L2_count;
extern const int _num__array_SpikeMonitor_L2_count;
extern int32_t *_array_SpikeMonitor_L2_N;
extern const int _num__array_SpikeMonitor_L2_N;
extern int32_t *_array_SpikeMonitor_L3__source_idx;
extern const int _num__array_SpikeMonitor_L3__source_idx;
extern int32_t *_array_SpikeMonitor_L3_count;
extern const int _num__array_SpikeMonitor_L3_count;
extern int32_t *_array_SpikeMonitor_L3_N;
extern const int _num__array_SpikeMonitor_L3_N;
extern int32_t *_array_StateMonitor_L1__indices;
extern const int _num__array_StateMonitor_L1__indices;
extern int32_t *_array_StateMonitor_L1_N;
extern const int _num__array_StateMonitor_L1_N;
extern double *_array_StateMonitor_L1_v;
extern const int _num__array_StateMonitor_L1_v;
extern int32_t *_array_StateMonitor_L2__indices;
extern const int _num__array_StateMonitor_L2__indices;
extern double *_array_StateMonitor_L2_g_e;
extern const int _num__array_StateMonitor_L2_g_e;
extern int32_t *_array_StateMonitor_L2_N;
extern const int _num__array_StateMonitor_L2_N;
extern double *_array_StateMonitor_L2_v;
extern const int _num__array_StateMonitor_L2_v;
extern int32_t *_array_StateMonitor_L3__indices;
extern const int _num__array_StateMonitor_L3__indices;
extern double *_array_StateMonitor_L3_g_e;
extern const int _num__array_StateMonitor_L3_g_e;
extern double *_array_StateMonitor_L3_g_i;
extern const int _num__array_StateMonitor_L3_g_i;
extern int32_t *_array_StateMonitor_L3_N;
extern const int _num__array_StateMonitor_L3_N;
extern double *_array_StateMonitor_L3_sum_w;
extern const int _num__array_StateMonitor_L3_sum_w;
extern double *_array_StateMonitor_L3_v;
extern const int _num__array_StateMonitor_L3_v;
extern double *_array_StateMonitor_L3_v_th;
extern const int _num__array_StateMonitor_L3_v_th;
extern int32_t *_array_StateMonitor_Syn23__indices;
extern const int _num__array_StateMonitor_Syn23__indices;
extern double *_array_StateMonitor_Syn23_c;
extern const int _num__array_StateMonitor_Syn23_c;
extern int32_t *_array_StateMonitor_Syn23_N;
extern const int _num__array_StateMonitor_Syn23_N;
extern double *_array_StateMonitor_Syn23_w;
extern const int _num__array_StateMonitor_Syn23_w;
extern double *_array_StateMonitor_Syn23_X;
extern const int _num__array_StateMonitor_Syn23_X;
extern int32_t *_array_Syn12_N;
extern const int _num__array_Syn12_N;
extern int32_t *_array_Syn12_sources;
extern const int _num__array_Syn12_sources;
extern int32_t *_array_Syn12_targets;
extern const int _num__array_Syn12_targets;
extern int32_t *_array_Syn23_N;
extern const int _num__array_Syn23_N;
extern int32_t *_array_Syn33_N;
extern const int _num__array_Syn33_N;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L1_v;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L2_g_e;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L2_v;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L3_g_e;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L3_g_i;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L3_sum_w;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L3_v;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_L3_v_th;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_c;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_w;
extern DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_X;

/////////////// static arrays /////////////
extern int32_t *_static_array__array_StateMonitor_L1__indices;
extern const int _num__static_array__array_StateMonitor_L1__indices;
extern int32_t *_static_array__array_StateMonitor_L2__indices;
extern const int _num__static_array__array_StateMonitor_L2__indices;
extern int32_t *_static_array__array_StateMonitor_L3__indices;
extern const int _num__static_array__array_StateMonitor_L3__indices;
extern int32_t *_static_array__array_StateMonitor_Syn23__indices;
extern const int _num__static_array__array_StateMonitor_Syn23__indices;
extern int32_t *_static_array__array_Syn12_sources;
extern const int _num__static_array__array_Syn12_sources;
extern int32_t *_static_array__array_Syn12_targets;
extern const int _num__static_array__array_Syn12_targets;
extern double *_timedarray_values;
extern const int _num__timedarray_values;

//////////////// synapses /////////////////
// Syn12
extern SynapticPathway Syn12_pre;
// Syn23
extern SynapticPathway Syn23_post;
extern SynapticPathway Syn23_pre;
// Syn33
extern SynapticPathway Syn33_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


