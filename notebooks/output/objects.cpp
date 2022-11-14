
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network network;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_L1__spikespace;
const int _num__array_L1__spikespace = 401;
int32_t * _array_L1_i;
const int _num__array_L1_i = 400;
double * _array_L1_lastspike;
const int _num__array_L1_lastspike = 400;
char * _array_L1_not_refractory;
const int _num__array_L1_not_refractory = 400;
double * _array_L1_v;
const int _num__array_L1_v = 400;
int32_t * _array_L2__spikespace;
const int _num__array_L2__spikespace = 21;
double * _array_L2_g_e;
const int _num__array_L2_g_e = 20;
int32_t * _array_L2_i;
const int _num__array_L2_i = 20;
double * _array_L2_lastspike;
const int _num__array_L2_lastspike = 20;
char * _array_L2_not_refractory;
const int _num__array_L2_not_refractory = 20;
double * _array_L2_sum_w;
const int _num__array_L2_sum_w = 20;
double * _array_L2_v;
const int _num__array_L2_v = 20;
int32_t * _array_L3__increase_thresholdspace;
const int _num__array_L3__increase_thresholdspace = 37;
int32_t * _array_L3__spikespace;
const int _num__array_L3__spikespace = 37;
double * _array_L3_g_e;
const int _num__array_L3_g_e = 36;
double * _array_L3_g_i;
const int _num__array_L3_g_i = 36;
int32_t * _array_L3_i;
const int _num__array_L3_i = 36;
char * _array_L3_is_winner;
const int _num__array_L3_is_winner = 36;
double * _array_L3_lastspike;
const int _num__array_L3_lastspike = 36;
char * _array_L3_not_refractory;
const int _num__array_L3_not_refractory = 36;
double * _array_L3_sum_w;
const int _num__array_L3_sum_w = 36;
double * _array_L3_v;
const int _num__array_L3_v = 36;
double * _array_L3_v_th;
const int _num__array_L3_v_th = 36;
int32_t * _array_SpikeMonitor_L1__source_idx;
const int _num__array_SpikeMonitor_L1__source_idx = 400;
int32_t * _array_SpikeMonitor_L1_count;
const int _num__array_SpikeMonitor_L1_count = 400;
int32_t * _array_SpikeMonitor_L1_N;
const int _num__array_SpikeMonitor_L1_N = 1;
int32_t * _array_SpikeMonitor_L2__source_idx;
const int _num__array_SpikeMonitor_L2__source_idx = 20;
int32_t * _array_SpikeMonitor_L2_count;
const int _num__array_SpikeMonitor_L2_count = 20;
int32_t * _array_SpikeMonitor_L2_N;
const int _num__array_SpikeMonitor_L2_N = 1;
int32_t * _array_SpikeMonitor_L3__source_idx;
const int _num__array_SpikeMonitor_L3__source_idx = 36;
int32_t * _array_SpikeMonitor_L3_count;
const int _num__array_SpikeMonitor_L3_count = 36;
int32_t * _array_SpikeMonitor_L3_N;
const int _num__array_SpikeMonitor_L3_N = 1;
int32_t * _array_StateMonitor_L1__indices;
const int _num__array_StateMonitor_L1__indices = 400;
int32_t * _array_StateMonitor_L1_N;
const int _num__array_StateMonitor_L1_N = 1;
double * _array_StateMonitor_L1_v;
const int _num__array_StateMonitor_L1_v = (0, 400);
int32_t * _array_StateMonitor_L2__indices;
const int _num__array_StateMonitor_L2__indices = 20;
double * _array_StateMonitor_L2_g_e;
const int _num__array_StateMonitor_L2_g_e = (0, 20);
int32_t * _array_StateMonitor_L2_N;
const int _num__array_StateMonitor_L2_N = 1;
double * _array_StateMonitor_L2_v;
const int _num__array_StateMonitor_L2_v = (0, 20);
int32_t * _array_StateMonitor_L3__indices;
const int _num__array_StateMonitor_L3__indices = 36;
double * _array_StateMonitor_L3_g_e;
const int _num__array_StateMonitor_L3_g_e = (0, 36);
double * _array_StateMonitor_L3_g_i;
const int _num__array_StateMonitor_L3_g_i = (0, 36);
int32_t * _array_StateMonitor_L3_N;
const int _num__array_StateMonitor_L3_N = 1;
double * _array_StateMonitor_L3_sum_w;
const int _num__array_StateMonitor_L3_sum_w = (0, 36);
double * _array_StateMonitor_L3_v;
const int _num__array_StateMonitor_L3_v = (0, 36);
double * _array_StateMonitor_L3_v_th;
const int _num__array_StateMonitor_L3_v_th = (0, 36);
int32_t * _array_StateMonitor_Syn23__indices;
const int _num__array_StateMonitor_Syn23__indices = 720;
double * _array_StateMonitor_Syn23_c;
const int _num__array_StateMonitor_Syn23_c = (0, 720);
int32_t * _array_StateMonitor_Syn23_N;
const int _num__array_StateMonitor_Syn23_N = 1;
double * _array_StateMonitor_Syn23_w;
const int _num__array_StateMonitor_Syn23_w = (0, 720);
double * _array_StateMonitor_Syn23_X;
const int _num__array_StateMonitor_Syn23_X = (0, 720);
int32_t * _array_Syn12_N;
const int _num__array_Syn12_N = 1;
int32_t * _array_Syn12_sources;
const int _num__array_Syn12_sources = 400;
int32_t * _array_Syn12_targets;
const int _num__array_Syn12_targets = 400;
int32_t * _array_Syn23_N;
const int _num__array_Syn23_N = 1;
int32_t * _array_Syn33_N;
const int _num__array_Syn33_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_SpikeMonitor_L1_i;
std::vector<double> _dynamic_array_SpikeMonitor_L1_t;
std::vector<int32_t> _dynamic_array_SpikeMonitor_L2_i;
std::vector<double> _dynamic_array_SpikeMonitor_L2_t;
std::vector<int32_t> _dynamic_array_SpikeMonitor_L3_i;
std::vector<double> _dynamic_array_SpikeMonitor_L3_t;
std::vector<double> _dynamic_array_StateMonitor_L1_t;
std::vector<double> _dynamic_array_StateMonitor_L2_t;
std::vector<double> _dynamic_array_StateMonitor_L3_t;
std::vector<double> _dynamic_array_StateMonitor_Syn23_t;
std::vector<int32_t> _dynamic_array_Syn12__synaptic_post;
std::vector<int32_t> _dynamic_array_Syn12__synaptic_pre;
std::vector<double> _dynamic_array_Syn12_delay;
std::vector<int32_t> _dynamic_array_Syn12_N_incoming;
std::vector<int32_t> _dynamic_array_Syn12_N_outgoing;
std::vector<double> _dynamic_array_Syn12_w;
std::vector<int32_t> _dynamic_array_Syn23__synaptic_post;
std::vector<int32_t> _dynamic_array_Syn23__synaptic_pre;
std::vector<double> _dynamic_array_Syn23_c;
std::vector<double> _dynamic_array_Syn23_count;
std::vector<double> _dynamic_array_Syn23_delay;
std::vector<double> _dynamic_array_Syn23_delay_1;
std::vector<int32_t> _dynamic_array_Syn23_N_incoming;
std::vector<int32_t> _dynamic_array_Syn23_N_outgoing;
std::vector<double> _dynamic_array_Syn23_X;
std::vector<double> _dynamic_array_Syn23_X_condition;
std::vector<int32_t> _dynamic_array_Syn33__synaptic_post;
std::vector<int32_t> _dynamic_array_Syn33__synaptic_pre;
std::vector<double> _dynamic_array_Syn33_delay;
std::vector<int32_t> _dynamic_array_Syn33_N_incoming;
std::vector<int32_t> _dynamic_array_Syn33_N_outgoing;
std::vector<double> _dynamic_array_Syn33_w;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_StateMonitor_L1_v;
DynamicArray2D<double> _dynamic_array_StateMonitor_L2_g_e;
DynamicArray2D<double> _dynamic_array_StateMonitor_L2_v;
DynamicArray2D<double> _dynamic_array_StateMonitor_L3_g_e;
DynamicArray2D<double> _dynamic_array_StateMonitor_L3_g_i;
DynamicArray2D<double> _dynamic_array_StateMonitor_L3_sum_w;
DynamicArray2D<double> _dynamic_array_StateMonitor_L3_v;
DynamicArray2D<double> _dynamic_array_StateMonitor_L3_v_th;
DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_c;
DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_w;
DynamicArray2D<double> _dynamic_array_StateMonitor_Syn23_X;

/////////////// static arrays /////////////
int32_t * _static_array__array_StateMonitor_L1__indices;
const int _num__static_array__array_StateMonitor_L1__indices = 400;
int32_t * _static_array__array_StateMonitor_L2__indices;
const int _num__static_array__array_StateMonitor_L2__indices = 20;
int32_t * _static_array__array_StateMonitor_L3__indices;
const int _num__static_array__array_StateMonitor_L3__indices = 36;
int32_t * _static_array__array_StateMonitor_Syn23__indices;
const int _num__static_array__array_StateMonitor_Syn23__indices = 720;
int32_t * _static_array__array_Syn12_sources;
const int _num__static_array__array_Syn12_sources = 400;
int32_t * _static_array__array_Syn12_targets;
const int _num__static_array__array_Syn12_targets = 400;
double * _timedarray_values;
const int _num__timedarray_values = 600800;

//////////////// synapses /////////////////
// Syn12
SynapticPathway Syn12_pre(
		_dynamic_array_Syn12__synaptic_pre,
		0, 400);
// Syn23
SynapticPathway Syn23_post(
		_dynamic_array_Syn23__synaptic_post,
		0, 36);
SynapticPathway Syn23_pre(
		_dynamic_array_Syn23__synaptic_pre,
		0, 20);
// Syn33
SynapticPathway Syn33_pre(
		_dynamic_array_Syn33__synaptic_pre,
		0, 36);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_L1__spikespace = new int32_t[401];
    
	for(int i=0; i<401; i++) _array_L1__spikespace[i] = 0;

	_array_L1_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_L1_i[i] = 0;

	_array_L1_lastspike = new double[400];
    
	for(int i=0; i<400; i++) _array_L1_lastspike[i] = 0;

	_array_L1_not_refractory = new char[400];
    
	for(int i=0; i<400; i++) _array_L1_not_refractory[i] = 0;

	_array_L1_v = new double[400];
    
	for(int i=0; i<400; i++) _array_L1_v[i] = 0;

	_array_L2__spikespace = new int32_t[21];
    
	for(int i=0; i<21; i++) _array_L2__spikespace[i] = 0;

	_array_L2_g_e = new double[20];
    
	for(int i=0; i<20; i++) _array_L2_g_e[i] = 0;

	_array_L2_i = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_L2_i[i] = 0;

	_array_L2_lastspike = new double[20];
    
	for(int i=0; i<20; i++) _array_L2_lastspike[i] = 0;

	_array_L2_not_refractory = new char[20];
    
	for(int i=0; i<20; i++) _array_L2_not_refractory[i] = 0;

	_array_L2_sum_w = new double[20];
    
	for(int i=0; i<20; i++) _array_L2_sum_w[i] = 0;

	_array_L2_v = new double[20];
    
	for(int i=0; i<20; i++) _array_L2_v[i] = 0;

	_array_L3__increase_thresholdspace = new int32_t[37];
    
	for(int i=0; i<37; i++) _array_L3__increase_thresholdspace[i] = 0;

	_array_L3__spikespace = new int32_t[37];
    
	for(int i=0; i<37; i++) _array_L3__spikespace[i] = 0;

	_array_L3_g_e = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_g_e[i] = 0;

	_array_L3_g_i = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_g_i[i] = 0;

	_array_L3_i = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_L3_i[i] = 0;

	_array_L3_is_winner = new char[36];
    
	for(int i=0; i<36; i++) _array_L3_is_winner[i] = 0;

	_array_L3_lastspike = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_lastspike[i] = 0;

	_array_L3_not_refractory = new char[36];
    
	for(int i=0; i<36; i++) _array_L3_not_refractory[i] = 0;

	_array_L3_sum_w = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_sum_w[i] = 0;

	_array_L3_v = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_v[i] = 0;

	_array_L3_v_th = new double[36];
    
	for(int i=0; i<36; i++) _array_L3_v_th[i] = 0;

	_array_SpikeMonitor_L1__source_idx = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_SpikeMonitor_L1__source_idx[i] = 0;

	_array_SpikeMonitor_L1_count = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_SpikeMonitor_L1_count[i] = 0;

	_array_SpikeMonitor_L1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_SpikeMonitor_L1_N[i] = 0;

	_array_SpikeMonitor_L2__source_idx = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_SpikeMonitor_L2__source_idx[i] = 0;

	_array_SpikeMonitor_L2_count = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_SpikeMonitor_L2_count[i] = 0;

	_array_SpikeMonitor_L2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_SpikeMonitor_L2_N[i] = 0;

	_array_SpikeMonitor_L3__source_idx = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_SpikeMonitor_L3__source_idx[i] = 0;

	_array_SpikeMonitor_L3_count = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_SpikeMonitor_L3_count[i] = 0;

	_array_SpikeMonitor_L3_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_SpikeMonitor_L3_N[i] = 0;

	_array_StateMonitor_L1__indices = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_StateMonitor_L1__indices[i] = 0;

	_array_StateMonitor_L1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_StateMonitor_L1_N[i] = 0;

	_array_StateMonitor_L2__indices = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_StateMonitor_L2__indices[i] = 0;

	_array_StateMonitor_L2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_StateMonitor_L2_N[i] = 0;

	_array_StateMonitor_L3__indices = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_StateMonitor_L3__indices[i] = 0;

	_array_StateMonitor_L3_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_StateMonitor_L3_N[i] = 0;

	_array_StateMonitor_Syn23__indices = new int32_t[720];
    
	for(int i=0; i<720; i++) _array_StateMonitor_Syn23__indices[i] = 0;

	_array_StateMonitor_Syn23_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_StateMonitor_Syn23_N[i] = 0;

	_array_Syn12_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_Syn12_N[i] = 0;

	_array_Syn12_sources = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_Syn12_sources[i] = 0;

	_array_Syn12_targets = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_Syn12_targets[i] = 0;

	_array_Syn23_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_Syn23_N[i] = 0;

	_array_Syn33_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_Syn33_N[i] = 0;


	// Arrays initialized to an "arange"
	_array_L1_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_L1_i[i] = 0 + i;

	_array_L2_i = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_L2_i[i] = 0 + i;

	_array_L3_i = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_L3_i[i] = 0 + i;

	_array_SpikeMonitor_L1__source_idx = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_SpikeMonitor_L1__source_idx[i] = 0 + i;

	_array_SpikeMonitor_L2__source_idx = new int32_t[20];
    
	for(int i=0; i<20; i++) _array_SpikeMonitor_L2__source_idx[i] = 0 + i;

	_array_SpikeMonitor_L3__source_idx = new int32_t[36];
    
	for(int i=0; i<36; i++) _array_SpikeMonitor_L3__source_idx[i] = 0 + i;


	// static arrays
	_static_array__array_StateMonitor_L1__indices = new int32_t[400];
	_static_array__array_StateMonitor_L2__indices = new int32_t[20];
	_static_array__array_StateMonitor_L3__indices = new int32_t[36];
	_static_array__array_StateMonitor_Syn23__indices = new int32_t[720];
	_static_array__array_Syn12_sources = new int32_t[400];
	_static_array__array_Syn12_targets = new int32_t[400];
	_timedarray_values = new double[600800];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_StateMonitor_L1__indices;
	f_static_array__array_StateMonitor_L1__indices.open("static_arrays/_static_array__array_StateMonitor_L1__indices", ios::in | ios::binary);
	if(f_static_array__array_StateMonitor_L1__indices.is_open())
	{
		f_static_array__array_StateMonitor_L1__indices.read(reinterpret_cast<char*>(_static_array__array_StateMonitor_L1__indices), 400*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_StateMonitor_L1__indices." << endl;
	}
	ifstream f_static_array__array_StateMonitor_L2__indices;
	f_static_array__array_StateMonitor_L2__indices.open("static_arrays/_static_array__array_StateMonitor_L2__indices", ios::in | ios::binary);
	if(f_static_array__array_StateMonitor_L2__indices.is_open())
	{
		f_static_array__array_StateMonitor_L2__indices.read(reinterpret_cast<char*>(_static_array__array_StateMonitor_L2__indices), 20*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_StateMonitor_L2__indices." << endl;
	}
	ifstream f_static_array__array_StateMonitor_L3__indices;
	f_static_array__array_StateMonitor_L3__indices.open("static_arrays/_static_array__array_StateMonitor_L3__indices", ios::in | ios::binary);
	if(f_static_array__array_StateMonitor_L3__indices.is_open())
	{
		f_static_array__array_StateMonitor_L3__indices.read(reinterpret_cast<char*>(_static_array__array_StateMonitor_L3__indices), 36*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_StateMonitor_L3__indices." << endl;
	}
	ifstream f_static_array__array_StateMonitor_Syn23__indices;
	f_static_array__array_StateMonitor_Syn23__indices.open("static_arrays/_static_array__array_StateMonitor_Syn23__indices", ios::in | ios::binary);
	if(f_static_array__array_StateMonitor_Syn23__indices.is_open())
	{
		f_static_array__array_StateMonitor_Syn23__indices.read(reinterpret_cast<char*>(_static_array__array_StateMonitor_Syn23__indices), 720*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_StateMonitor_Syn23__indices." << endl;
	}
	ifstream f_static_array__array_Syn12_sources;
	f_static_array__array_Syn12_sources.open("static_arrays/_static_array__array_Syn12_sources", ios::in | ios::binary);
	if(f_static_array__array_Syn12_sources.is_open())
	{
		f_static_array__array_Syn12_sources.read(reinterpret_cast<char*>(_static_array__array_Syn12_sources), 400*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_Syn12_sources." << endl;
	}
	ifstream f_static_array__array_Syn12_targets;
	f_static_array__array_Syn12_targets.open("static_arrays/_static_array__array_Syn12_targets", ios::in | ios::binary);
	if(f_static_array__array_Syn12_targets.is_open())
	{
		f_static_array__array_Syn12_targets.read(reinterpret_cast<char*>(_static_array__array_Syn12_targets), 400*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_Syn12_targets." << endl;
	}
	ifstream f_timedarray_values;
	f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
	if(f_timedarray_values.is_open())
	{
		f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 600800*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _timedarray_values." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_1978099143", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_2669362164", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_144223508", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_L1__spikespace;
	outfile__array_L1__spikespace.open("results/_array_L1__spikespace_475383852", ios::binary | ios::out);
	if(outfile__array_L1__spikespace.is_open())
	{
		outfile__array_L1__spikespace.write(reinterpret_cast<char*>(_array_L1__spikespace), 401*sizeof(_array_L1__spikespace[0]));
		outfile__array_L1__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_L1__spikespace." << endl;
	}
	ofstream outfile__array_L1_i;
	outfile__array_L1_i.open("results/_array_L1_i_3360504313", ios::binary | ios::out);
	if(outfile__array_L1_i.is_open())
	{
		outfile__array_L1_i.write(reinterpret_cast<char*>(_array_L1_i), 400*sizeof(_array_L1_i[0]));
		outfile__array_L1_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_L1_i." << endl;
	}
	ofstream outfile__array_L1_lastspike;
	outfile__array_L1_lastspike.open("results/_array_L1_lastspike_3760261053", ios::binary | ios::out);
	if(outfile__array_L1_lastspike.is_open())
	{
		outfile__array_L1_lastspike.write(reinterpret_cast<char*>(_array_L1_lastspike), 400*sizeof(_array_L1_lastspike[0]));
		outfile__array_L1_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_L1_lastspike." << endl;
	}
	ofstream outfile__array_L1_not_refractory;
	outfile__array_L1_not_refractory.open("results/_array_L1_not_refractory_3926422587", ios::binary | ios::out);
	if(outfile__array_L1_not_refractory.is_open())
	{
		outfile__array_L1_not_refractory.write(reinterpret_cast<char*>(_array_L1_not_refractory), 400*sizeof(_array_L1_not_refractory[0]));
		outfile__array_L1_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_L1_not_refractory." << endl;
	}
	ofstream outfile__array_L1_v;
	outfile__array_L1_v.open("results/_array_L1_v_1162163212", ios::binary | ios::out);
	if(outfile__array_L1_v.is_open())
	{
		outfile__array_L1_v.write(reinterpret_cast<char*>(_array_L1_v), 400*sizeof(_array_L1_v[0]));
		outfile__array_L1_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_L1_v." << endl;
	}
	ofstream outfile__array_L2__spikespace;
	outfile__array_L2__spikespace.open("results/_array_L2__spikespace_2711594210", ios::binary | ios::out);
	if(outfile__array_L2__spikespace.is_open())
	{
		outfile__array_L2__spikespace.write(reinterpret_cast<char*>(_array_L2__spikespace), 21*sizeof(_array_L2__spikespace[0]));
		outfile__array_L2__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2__spikespace." << endl;
	}
	ofstream outfile__array_L2_g_e;
	outfile__array_L2_g_e.open("results/_array_L2_g_e_2523368408", ios::binary | ios::out);
	if(outfile__array_L2_g_e.is_open())
	{
		outfile__array_L2_g_e.write(reinterpret_cast<char*>(_array_L2_g_e), 20*sizeof(_array_L2_g_e[0]));
		outfile__array_L2_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_g_e." << endl;
	}
	ofstream outfile__array_L2_i;
	outfile__array_L2_i.open("results/_array_L2_i_3389753248", ios::binary | ios::out);
	if(outfile__array_L2_i.is_open())
	{
		outfile__array_L2_i.write(reinterpret_cast<char*>(_array_L2_i), 20*sizeof(_array_L2_i[0]));
		outfile__array_L2_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_i." << endl;
	}
	ofstream outfile__array_L2_lastspike;
	outfile__array_L2_lastspike.open("results/_array_L2_lastspike_2042847676", ios::binary | ios::out);
	if(outfile__array_L2_lastspike.is_open())
	{
		outfile__array_L2_lastspike.write(reinterpret_cast<char*>(_array_L2_lastspike), 20*sizeof(_array_L2_lastspike[0]));
		outfile__array_L2_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_lastspike." << endl;
	}
	ofstream outfile__array_L2_not_refractory;
	outfile__array_L2_not_refractory.open("results/_array_L2_not_refractory_3284189385", ios::binary | ios::out);
	if(outfile__array_L2_not_refractory.is_open())
	{
		outfile__array_L2_not_refractory.write(reinterpret_cast<char*>(_array_L2_not_refractory), 20*sizeof(_array_L2_not_refractory[0]));
		outfile__array_L2_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_not_refractory." << endl;
	}
	ofstream outfile__array_L2_sum_w;
	outfile__array_L2_sum_w.open("results/_array_L2_sum_w_3589020547", ios::binary | ios::out);
	if(outfile__array_L2_sum_w.is_open())
	{
		outfile__array_L2_sum_w.write(reinterpret_cast<char*>(_array_L2_sum_w), 20*sizeof(_array_L2_sum_w[0]));
		outfile__array_L2_sum_w.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_sum_w." << endl;
	}
	ofstream outfile__array_L2_v;
	outfile__array_L2_v.open("results/_array_L2_v_1191414357", ios::binary | ios::out);
	if(outfile__array_L2_v.is_open())
	{
		outfile__array_L2_v.write(reinterpret_cast<char*>(_array_L2_v), 20*sizeof(_array_L2_v[0]));
		outfile__array_L2_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_L2_v." << endl;
	}
	ofstream outfile__array_L3__increase_thresholdspace;
	outfile__array_L3__increase_thresholdspace.open("results/_array_L3__increase_thresholdspace_2791376074", ios::binary | ios::out);
	if(outfile__array_L3__increase_thresholdspace.is_open())
	{
		outfile__array_L3__increase_thresholdspace.write(reinterpret_cast<char*>(_array_L3__increase_thresholdspace), 37*sizeof(_array_L3__increase_thresholdspace[0]));
		outfile__array_L3__increase_thresholdspace.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3__increase_thresholdspace." << endl;
	}
	ofstream outfile__array_L3__spikespace;
	outfile__array_L3__spikespace.open("results/_array_L3__spikespace_2080996711", ios::binary | ios::out);
	if(outfile__array_L3__spikespace.is_open())
	{
		outfile__array_L3__spikespace.write(reinterpret_cast<char*>(_array_L3__spikespace), 37*sizeof(_array_L3__spikespace[0]));
		outfile__array_L3__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3__spikespace." << endl;
	}
	ofstream outfile__array_L3_g_e;
	outfile__array_L3_g_e.open("results/_array_L3_g_e_2869404264", ios::binary | ios::out);
	if(outfile__array_L3_g_e.is_open())
	{
		outfile__array_L3_g_e.write(reinterpret_cast<char*>(_array_L3_g_e), 36*sizeof(_array_L3_g_e[0]));
		outfile__array_L3_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_g_e." << endl;
	}
	ofstream outfile__array_L3_g_i;
	outfile__array_L3_g_i.open("results/_array_L3_g_i_2729569859", ios::binary | ios::out);
	if(outfile__array_L3_g_i.is_open())
	{
		outfile__array_L3_g_i.write(reinterpret_cast<char*>(_array_L3_g_i), 36*sizeof(_array_L3_g_i[0]));
		outfile__array_L3_g_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_g_i." << endl;
	}
	ofstream outfile__array_L3_i;
	outfile__array_L3_i.open("results/_array_L3_i_3419008407", ios::binary | ios::out);
	if(outfile__array_L3_i.is_open())
	{
		outfile__array_L3_i.write(reinterpret_cast<char*>(_array_L3_i), 36*sizeof(_array_L3_i[0]));
		outfile__array_L3_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_i." << endl;
	}
	ofstream outfile__array_L3_is_winner;
	outfile__array_L3_is_winner.open("results/_array_L3_is_winner_3225746784", ios::binary | ios::out);
	if(outfile__array_L3_is_winner.is_open())
	{
		outfile__array_L3_is_winner.write(reinterpret_cast<char*>(_array_L3_is_winner), 36*sizeof(_array_L3_is_winner[0]));
		outfile__array_L3_is_winner.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_is_winner." << endl;
	}
	ofstream outfile__array_L3_lastspike;
	outfile__array_L3_lastspike.open("results/_array_L3_lastspike_3092102780", ios::binary | ios::out);
	if(outfile__array_L3_lastspike.is_open())
	{
		outfile__array_L3_lastspike.write(reinterpret_cast<char*>(_array_L3_lastspike), 36*sizeof(_array_L3_lastspike[0]));
		outfile__array_L3_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_lastspike." << endl;
	}
	ofstream outfile__array_L3_not_refractory;
	outfile__array_L3_not_refractory.open("results/_array_L3_not_refractory_1839747416", ios::binary | ios::out);
	if(outfile__array_L3_not_refractory.is_open())
	{
		outfile__array_L3_not_refractory.write(reinterpret_cast<char*>(_array_L3_not_refractory), 36*sizeof(_array_L3_not_refractory[0]));
		outfile__array_L3_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_not_refractory." << endl;
	}
	ofstream outfile__array_L3_sum_w;
	outfile__array_L3_sum_w.open("results/_array_L3_sum_w_1939542071", ios::binary | ios::out);
	if(outfile__array_L3_sum_w.is_open())
	{
		outfile__array_L3_sum_w.write(reinterpret_cast<char*>(_array_L3_sum_w), 36*sizeof(_array_L3_sum_w[0]));
		outfile__array_L3_sum_w.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_sum_w." << endl;
	}
	ofstream outfile__array_L3_v;
	outfile__array_L3_v.open("results/_array_L3_v_1187111010", ios::binary | ios::out);
	if(outfile__array_L3_v.is_open())
	{
		outfile__array_L3_v.write(reinterpret_cast<char*>(_array_L3_v), 36*sizeof(_array_L3_v[0]));
		outfile__array_L3_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_v." << endl;
	}
	ofstream outfile__array_L3_v_th;
	outfile__array_L3_v_th.open("results/_array_L3_v_th_1775631301", ios::binary | ios::out);
	if(outfile__array_L3_v_th.is_open())
	{
		outfile__array_L3_v_th.write(reinterpret_cast<char*>(_array_L3_v_th), 36*sizeof(_array_L3_v_th[0]));
		outfile__array_L3_v_th.close();
	} else
	{
		std::cout << "Error writing output file for _array_L3_v_th." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L1__source_idx;
	outfile__array_SpikeMonitor_L1__source_idx.open("results/_array_SpikeMonitor_L1__source_idx_2147007574", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L1__source_idx.is_open())
	{
		outfile__array_SpikeMonitor_L1__source_idx.write(reinterpret_cast<char*>(_array_SpikeMonitor_L1__source_idx), 400*sizeof(_array_SpikeMonitor_L1__source_idx[0]));
		outfile__array_SpikeMonitor_L1__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L1__source_idx." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L1_count;
	outfile__array_SpikeMonitor_L1_count.open("results/_array_SpikeMonitor_L1_count_1780477856", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L1_count.is_open())
	{
		outfile__array_SpikeMonitor_L1_count.write(reinterpret_cast<char*>(_array_SpikeMonitor_L1_count), 400*sizeof(_array_SpikeMonitor_L1_count[0]));
		outfile__array_SpikeMonitor_L1_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L1_count." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L1_N;
	outfile__array_SpikeMonitor_L1_N.open("results/_array_SpikeMonitor_L1_N_2198786965", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L1_N.is_open())
	{
		outfile__array_SpikeMonitor_L1_N.write(reinterpret_cast<char*>(_array_SpikeMonitor_L1_N), 1*sizeof(_array_SpikeMonitor_L1_N[0]));
		outfile__array_SpikeMonitor_L1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L1_N." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L2__source_idx;
	outfile__array_SpikeMonitor_L2__source_idx.open("results/_array_SpikeMonitor_L2__source_idx_3258110104", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L2__source_idx.is_open())
	{
		outfile__array_SpikeMonitor_L2__source_idx.write(reinterpret_cast<char*>(_array_SpikeMonitor_L2__source_idx), 20*sizeof(_array_SpikeMonitor_L2__source_idx[0]));
		outfile__array_SpikeMonitor_L2__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L2__source_idx." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L2_count;
	outfile__array_SpikeMonitor_L2_count.open("results/_array_SpikeMonitor_L2_count_1542976829", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L2_count.is_open())
	{
		outfile__array_SpikeMonitor_L2_count.write(reinterpret_cast<char*>(_array_SpikeMonitor_L2_count), 20*sizeof(_array_SpikeMonitor_L2_count[0]));
		outfile__array_SpikeMonitor_L2_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L2_count." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L2_N;
	outfile__array_SpikeMonitor_L2_N.open("results/_array_SpikeMonitor_L2_N_2169007564", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L2_N.is_open())
	{
		outfile__array_SpikeMonitor_L2_N.write(reinterpret_cast<char*>(_array_SpikeMonitor_L2_N), 1*sizeof(_array_SpikeMonitor_L2_N[0]));
		outfile__array_SpikeMonitor_L2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L2_N." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L3__source_idx;
	outfile__array_SpikeMonitor_L3__source_idx.open("results/_array_SpikeMonitor_L3__source_idx_530843933", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L3__source_idx.is_open())
	{
		outfile__array_SpikeMonitor_L3__source_idx.write(reinterpret_cast<char*>(_array_SpikeMonitor_L3__source_idx), 36*sizeof(_array_SpikeMonitor_L3__source_idx[0]));
		outfile__array_SpikeMonitor_L3__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L3__source_idx." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L3_count;
	outfile__array_SpikeMonitor_L3_count.open("results/_array_SpikeMonitor_L3_count_4253089417", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L3_count.is_open())
	{
		outfile__array_SpikeMonitor_L3_count.write(reinterpret_cast<char*>(_array_SpikeMonitor_L3_count), 36*sizeof(_array_SpikeMonitor_L3_count[0]));
		outfile__array_SpikeMonitor_L3_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L3_count." << endl;
	}
	ofstream outfile__array_SpikeMonitor_L3_N;
	outfile__array_SpikeMonitor_L3_N.open("results/_array_SpikeMonitor_L3_N_2156529659", ios::binary | ios::out);
	if(outfile__array_SpikeMonitor_L3_N.is_open())
	{
		outfile__array_SpikeMonitor_L3_N.write(reinterpret_cast<char*>(_array_SpikeMonitor_L3_N), 1*sizeof(_array_SpikeMonitor_L3_N[0]));
		outfile__array_SpikeMonitor_L3_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_SpikeMonitor_L3_N." << endl;
	}
	ofstream outfile__array_StateMonitor_L1__indices;
	outfile__array_StateMonitor_L1__indices.open("results/_array_StateMonitor_L1__indices_1733206065", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L1__indices.is_open())
	{
		outfile__array_StateMonitor_L1__indices.write(reinterpret_cast<char*>(_array_StateMonitor_L1__indices), 400*sizeof(_array_StateMonitor_L1__indices[0]));
		outfile__array_StateMonitor_L1__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L1__indices." << endl;
	}
	ofstream outfile__array_StateMonitor_L1_N;
	outfile__array_StateMonitor_L1_N.open("results/_array_StateMonitor_L1_N_2828065551", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L1_N.is_open())
	{
		outfile__array_StateMonitor_L1_N.write(reinterpret_cast<char*>(_array_StateMonitor_L1_N), 1*sizeof(_array_StateMonitor_L1_N[0]));
		outfile__array_StateMonitor_L1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L1_N." << endl;
	}
	ofstream outfile__array_StateMonitor_L2__indices;
	outfile__array_StateMonitor_L2__indices.open("results/_array_StateMonitor_L2__indices_2356747058", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L2__indices.is_open())
	{
		outfile__array_StateMonitor_L2__indices.write(reinterpret_cast<char*>(_array_StateMonitor_L2__indices), 20*sizeof(_array_StateMonitor_L2__indices[0]));
		outfile__array_StateMonitor_L2__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L2__indices." << endl;
	}
	ofstream outfile__array_StateMonitor_L2_N;
	outfile__array_StateMonitor_L2_N.open("results/_array_StateMonitor_L2_N_2866177366", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L2_N.is_open())
	{
		outfile__array_StateMonitor_L2_N.write(reinterpret_cast<char*>(_array_StateMonitor_L2_N), 1*sizeof(_array_StateMonitor_L2_N[0]));
		outfile__array_StateMonitor_L2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L2_N." << endl;
	}
	ofstream outfile__array_StateMonitor_L3__indices;
	outfile__array_StateMonitor_L3__indices.open("results/_array_StateMonitor_L3__indices_1673228300", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L3__indices.is_open())
	{
		outfile__array_StateMonitor_L3__indices.write(reinterpret_cast<char*>(_array_StateMonitor_L3__indices), 36*sizeof(_array_StateMonitor_L3__indices[0]));
		outfile__array_StateMonitor_L3__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L3__indices." << endl;
	}
	ofstream outfile__array_StateMonitor_L3_N;
	outfile__array_StateMonitor_L3_N.open("results/_array_StateMonitor_L3_N_2870218593", ios::binary | ios::out);
	if(outfile__array_StateMonitor_L3_N.is_open())
	{
		outfile__array_StateMonitor_L3_N.write(reinterpret_cast<char*>(_array_StateMonitor_L3_N), 1*sizeof(_array_StateMonitor_L3_N[0]));
		outfile__array_StateMonitor_L3_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_L3_N." << endl;
	}
	ofstream outfile__array_StateMonitor_Syn23__indices;
	outfile__array_StateMonitor_Syn23__indices.open("results/_array_StateMonitor_Syn23__indices_1283008279", ios::binary | ios::out);
	if(outfile__array_StateMonitor_Syn23__indices.is_open())
	{
		outfile__array_StateMonitor_Syn23__indices.write(reinterpret_cast<char*>(_array_StateMonitor_Syn23__indices), 720*sizeof(_array_StateMonitor_Syn23__indices[0]));
		outfile__array_StateMonitor_Syn23__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_Syn23__indices." << endl;
	}
	ofstream outfile__array_StateMonitor_Syn23_N;
	outfile__array_StateMonitor_Syn23_N.open("results/_array_StateMonitor_Syn23_N_595693030", ios::binary | ios::out);
	if(outfile__array_StateMonitor_Syn23_N.is_open())
	{
		outfile__array_StateMonitor_Syn23_N.write(reinterpret_cast<char*>(_array_StateMonitor_Syn23_N), 1*sizeof(_array_StateMonitor_Syn23_N[0]));
		outfile__array_StateMonitor_Syn23_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_StateMonitor_Syn23_N." << endl;
	}
	ofstream outfile__array_Syn12_N;
	outfile__array_Syn12_N.open("results/_array_Syn12_N_678490045", ios::binary | ios::out);
	if(outfile__array_Syn12_N.is_open())
	{
		outfile__array_Syn12_N.write(reinterpret_cast<char*>(_array_Syn12_N), 1*sizeof(_array_Syn12_N[0]));
		outfile__array_Syn12_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_Syn12_N." << endl;
	}
	ofstream outfile__array_Syn12_sources;
	outfile__array_Syn12_sources.open("results/_array_Syn12_sources_3145134698", ios::binary | ios::out);
	if(outfile__array_Syn12_sources.is_open())
	{
		outfile__array_Syn12_sources.write(reinterpret_cast<char*>(_array_Syn12_sources), 400*sizeof(_array_Syn12_sources[0]));
		outfile__array_Syn12_sources.close();
	} else
	{
		std::cout << "Error writing output file for _array_Syn12_sources." << endl;
	}
	ofstream outfile__array_Syn12_targets;
	outfile__array_Syn12_targets.open("results/_array_Syn12_targets_3328739723", ios::binary | ios::out);
	if(outfile__array_Syn12_targets.is_open())
	{
		outfile__array_Syn12_targets.write(reinterpret_cast<char*>(_array_Syn12_targets), 400*sizeof(_array_Syn12_targets[0]));
		outfile__array_Syn12_targets.close();
	} else
	{
		std::cout << "Error writing output file for _array_Syn12_targets." << endl;
	}
	ofstream outfile__array_Syn23_N;
	outfile__array_Syn23_N.open("results/_array_Syn23_N_990325348", ios::binary | ios::out);
	if(outfile__array_Syn23_N.is_open())
	{
		outfile__array_Syn23_N.write(reinterpret_cast<char*>(_array_Syn23_N), 1*sizeof(_array_Syn23_N[0]));
		outfile__array_Syn23_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_Syn23_N." << endl;
	}
	ofstream outfile__array_Syn33_N;
	outfile__array_Syn33_N.open("results/_array_Syn33_N_2210090241", ios::binary | ios::out);
	if(outfile__array_Syn33_N.is_open())
	{
		outfile__array_Syn33_N.write(reinterpret_cast<char*>(_array_Syn33_N), 1*sizeof(_array_Syn33_N[0]));
		outfile__array_Syn33_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_Syn33_N." << endl;
	}

	ofstream outfile__dynamic_array_SpikeMonitor_L1_i;
	outfile__dynamic_array_SpikeMonitor_L1_i.open("results/_dynamic_array_SpikeMonitor_L1_i_3235088634", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L1_i.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L1_i.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L1_i.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L1_i[0]), _dynamic_array_SpikeMonitor_L1_i.size()*sizeof(_dynamic_array_SpikeMonitor_L1_i[0]));
		    outfile__dynamic_array_SpikeMonitor_L1_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L1_i." << endl;
	}
	ofstream outfile__dynamic_array_SpikeMonitor_L1_t;
	outfile__dynamic_array_SpikeMonitor_L1_t.open("results/_dynamic_array_SpikeMonitor_L1_t_2748703779", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L1_t.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L1_t.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L1_t.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L1_t[0]), _dynamic_array_SpikeMonitor_L1_t.size()*sizeof(_dynamic_array_SpikeMonitor_L1_t[0]));
		    outfile__dynamic_array_SpikeMonitor_L1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L1_t." << endl;
	}
	ofstream outfile__dynamic_array_SpikeMonitor_L2_i;
	outfile__dynamic_array_SpikeMonitor_L2_i.open("results/_dynamic_array_SpikeMonitor_L2_i_3264558755", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L2_i.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L2_i.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L2_i.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L2_i[0]), _dynamic_array_SpikeMonitor_L2_i.size()*sizeof(_dynamic_array_SpikeMonitor_L2_i[0]));
		    outfile__dynamic_array_SpikeMonitor_L2_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L2_i." << endl;
	}
	ofstream outfile__dynamic_array_SpikeMonitor_L2_t;
	outfile__dynamic_array_SpikeMonitor_L2_t.open("results/_dynamic_array_SpikeMonitor_L2_t_2710788730", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L2_t.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L2_t.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L2_t.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L2_t[0]), _dynamic_array_SpikeMonitor_L2_t.size()*sizeof(_dynamic_array_SpikeMonitor_L2_t[0]));
		    outfile__dynamic_array_SpikeMonitor_L2_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L2_t." << endl;
	}
	ofstream outfile__dynamic_array_SpikeMonitor_L3_i;
	outfile__dynamic_array_SpikeMonitor_L3_i.open("results/_dynamic_array_SpikeMonitor_L3_i_3277282452", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L3_i.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L3_i.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L3_i.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L3_i[0]), _dynamic_array_SpikeMonitor_L3_i.size()*sizeof(_dynamic_array_SpikeMonitor_L3_i[0]));
		    outfile__dynamic_array_SpikeMonitor_L3_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L3_i." << endl;
	}
	ofstream outfile__dynamic_array_SpikeMonitor_L3_t;
	outfile__dynamic_array_SpikeMonitor_L3_t.open("results/_dynamic_array_SpikeMonitor_L3_t_2689675341", ios::binary | ios::out);
	if(outfile__dynamic_array_SpikeMonitor_L3_t.is_open())
	{
        if (! _dynamic_array_SpikeMonitor_L3_t.empty() )
        {
			outfile__dynamic_array_SpikeMonitor_L3_t.write(reinterpret_cast<char*>(&_dynamic_array_SpikeMonitor_L3_t[0]), _dynamic_array_SpikeMonitor_L3_t.size()*sizeof(_dynamic_array_SpikeMonitor_L3_t[0]));
		    outfile__dynamic_array_SpikeMonitor_L3_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_SpikeMonitor_L3_t." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L1_t;
	outfile__dynamic_array_StateMonitor_L1_t.open("results/_dynamic_array_StateMonitor_L1_t_2286677177", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L1_t.is_open())
	{
        if (! _dynamic_array_StateMonitor_L1_t.empty() )
        {
			outfile__dynamic_array_StateMonitor_L1_t.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L1_t[0]), _dynamic_array_StateMonitor_L1_t.size()*sizeof(_dynamic_array_StateMonitor_L1_t[0]));
		    outfile__dynamic_array_StateMonitor_L1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L1_t." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L2_t;
	outfile__dynamic_array_StateMonitor_L2_t.open("results/_dynamic_array_StateMonitor_L2_t_2316128992", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L2_t.is_open())
	{
        if (! _dynamic_array_StateMonitor_L2_t.empty() )
        {
			outfile__dynamic_array_StateMonitor_L2_t.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L2_t[0]), _dynamic_array_StateMonitor_L2_t.size()*sizeof(_dynamic_array_StateMonitor_L2_t[0]));
		    outfile__dynamic_array_StateMonitor_L2_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L2_t." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_t;
	outfile__dynamic_array_StateMonitor_L3_t.open("results/_dynamic_array_StateMonitor_L3_t_2345613527", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_t.is_open())
	{
        if (! _dynamic_array_StateMonitor_L3_t.empty() )
        {
			outfile__dynamic_array_StateMonitor_L3_t.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_t[0]), _dynamic_array_StateMonitor_L3_t.size()*sizeof(_dynamic_array_StateMonitor_L3_t[0]));
		    outfile__dynamic_array_StateMonitor_L3_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_t." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_Syn23_t;
	outfile__dynamic_array_StateMonitor_Syn23_t.open("results/_dynamic_array_StateMonitor_Syn23_t_986604923", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_Syn23_t.is_open())
	{
        if (! _dynamic_array_StateMonitor_Syn23_t.empty() )
        {
			outfile__dynamic_array_StateMonitor_Syn23_t.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_Syn23_t[0]), _dynamic_array_StateMonitor_Syn23_t.size()*sizeof(_dynamic_array_StateMonitor_Syn23_t[0]));
		    outfile__dynamic_array_StateMonitor_Syn23_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_Syn23_t." << endl;
	}
	ofstream outfile__dynamic_array_Syn12__synaptic_post;
	outfile__dynamic_array_Syn12__synaptic_post.open("results/_dynamic_array_Syn12__synaptic_post_1480340758", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12__synaptic_post.is_open())
	{
        if (! _dynamic_array_Syn12__synaptic_post.empty() )
        {
			outfile__dynamic_array_Syn12__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_Syn12__synaptic_post[0]), _dynamic_array_Syn12__synaptic_post.size()*sizeof(_dynamic_array_Syn12__synaptic_post[0]));
		    outfile__dynamic_array_Syn12__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_Syn12__synaptic_pre;
	outfile__dynamic_array_Syn12__synaptic_pre.open("results/_dynamic_array_Syn12__synaptic_pre_1353428514", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12__synaptic_pre.is_open())
	{
        if (! _dynamic_array_Syn12__synaptic_pre.empty() )
        {
			outfile__dynamic_array_Syn12__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_Syn12__synaptic_pre[0]), _dynamic_array_Syn12__synaptic_pre.size()*sizeof(_dynamic_array_Syn12__synaptic_pre[0]));
		    outfile__dynamic_array_Syn12__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_Syn12_delay;
	outfile__dynamic_array_Syn12_delay.open("results/_dynamic_array_Syn12_delay_2706255449", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12_delay.is_open())
	{
        if (! _dynamic_array_Syn12_delay.empty() )
        {
			outfile__dynamic_array_Syn12_delay.write(reinterpret_cast<char*>(&_dynamic_array_Syn12_delay[0]), _dynamic_array_Syn12_delay.size()*sizeof(_dynamic_array_Syn12_delay[0]));
		    outfile__dynamic_array_Syn12_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12_delay." << endl;
	}
	ofstream outfile__dynamic_array_Syn12_N_incoming;
	outfile__dynamic_array_Syn12_N_incoming.open("results/_dynamic_array_Syn12_N_incoming_1020372178", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12_N_incoming.is_open())
	{
        if (! _dynamic_array_Syn12_N_incoming.empty() )
        {
			outfile__dynamic_array_Syn12_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_Syn12_N_incoming[0]), _dynamic_array_Syn12_N_incoming.size()*sizeof(_dynamic_array_Syn12_N_incoming[0]));
		    outfile__dynamic_array_Syn12_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_Syn12_N_outgoing;
	outfile__dynamic_array_Syn12_N_outgoing.open("results/_dynamic_array_Syn12_N_outgoing_466473992", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12_N_outgoing.is_open())
	{
        if (! _dynamic_array_Syn12_N_outgoing.empty() )
        {
			outfile__dynamic_array_Syn12_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_Syn12_N_outgoing[0]), _dynamic_array_Syn12_N_outgoing.size()*sizeof(_dynamic_array_Syn12_N_outgoing[0]));
		    outfile__dynamic_array_Syn12_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_Syn12_w;
	outfile__dynamic_array_Syn12_w.open("results/_dynamic_array_Syn12_w_3592466126", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn12_w.is_open())
	{
        if (! _dynamic_array_Syn12_w.empty() )
        {
			outfile__dynamic_array_Syn12_w.write(reinterpret_cast<char*>(&_dynamic_array_Syn12_w[0]), _dynamic_array_Syn12_w.size()*sizeof(_dynamic_array_Syn12_w[0]));
		    outfile__dynamic_array_Syn12_w.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn12_w." << endl;
	}
	ofstream outfile__dynamic_array_Syn23__synaptic_post;
	outfile__dynamic_array_Syn23__synaptic_post.open("results/_dynamic_array_Syn23__synaptic_post_2781806339", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23__synaptic_post.is_open())
	{
        if (! _dynamic_array_Syn23__synaptic_post.empty() )
        {
			outfile__dynamic_array_Syn23__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_Syn23__synaptic_post[0]), _dynamic_array_Syn23__synaptic_post.size()*sizeof(_dynamic_array_Syn23__synaptic_post[0]));
		    outfile__dynamic_array_Syn23__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_Syn23__synaptic_pre;
	outfile__dynamic_array_Syn23__synaptic_pre.open("results/_dynamic_array_Syn23__synaptic_pre_3231236408", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23__synaptic_pre.is_open())
	{
        if (! _dynamic_array_Syn23__synaptic_pre.empty() )
        {
			outfile__dynamic_array_Syn23__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_Syn23__synaptic_pre[0]), _dynamic_array_Syn23__synaptic_pre.size()*sizeof(_dynamic_array_Syn23__synaptic_pre[0]));
		    outfile__dynamic_array_Syn23__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_c;
	outfile__dynamic_array_Syn23_c.open("results/_dynamic_array_Syn23_c_3750608746", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_c.is_open())
	{
        if (! _dynamic_array_Syn23_c.empty() )
        {
			outfile__dynamic_array_Syn23_c.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_c[0]), _dynamic_array_Syn23_c.size()*sizeof(_dynamic_array_Syn23_c[0]));
		    outfile__dynamic_array_Syn23_c.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_c." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_count;
	outfile__dynamic_array_Syn23_count.open("results/_dynamic_array_Syn23_count_3203786487", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_count.is_open())
	{
        if (! _dynamic_array_Syn23_count.empty() )
        {
			outfile__dynamic_array_Syn23_count.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_count[0]), _dynamic_array_Syn23_count.size()*sizeof(_dynamic_array_Syn23_count[0]));
		    outfile__dynamic_array_Syn23_count.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_count." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_delay;
	outfile__dynamic_array_Syn23_delay.open("results/_dynamic_array_Syn23_delay_2310414862", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_delay.is_open())
	{
        if (! _dynamic_array_Syn23_delay.empty() )
        {
			outfile__dynamic_array_Syn23_delay.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_delay[0]), _dynamic_array_Syn23_delay.size()*sizeof(_dynamic_array_Syn23_delay[0]));
		    outfile__dynamic_array_Syn23_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_delay." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_delay_1;
	outfile__dynamic_array_Syn23_delay_1.open("results/_dynamic_array_Syn23_delay_1_1724364418", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_delay_1.is_open())
	{
        if (! _dynamic_array_Syn23_delay_1.empty() )
        {
			outfile__dynamic_array_Syn23_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_delay_1[0]), _dynamic_array_Syn23_delay_1.size()*sizeof(_dynamic_array_Syn23_delay_1[0]));
		    outfile__dynamic_array_Syn23_delay_1.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_delay_1." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_N_incoming;
	outfile__dynamic_array_Syn23_N_incoming.open("results/_dynamic_array_Syn23_N_incoming_448693363", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_N_incoming.is_open())
	{
        if (! _dynamic_array_Syn23_N_incoming.empty() )
        {
			outfile__dynamic_array_Syn23_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_N_incoming[0]), _dynamic_array_Syn23_N_incoming.size()*sizeof(_dynamic_array_Syn23_N_incoming[0]));
		    outfile__dynamic_array_Syn23_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_N_outgoing;
	outfile__dynamic_array_Syn23_N_outgoing.open("results/_dynamic_array_Syn23_N_outgoing_1034089641", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_N_outgoing.is_open())
	{
        if (! _dynamic_array_Syn23_N_outgoing.empty() )
        {
			outfile__dynamic_array_Syn23_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_N_outgoing[0]), _dynamic_array_Syn23_N_outgoing.size()*sizeof(_dynamic_array_Syn23_N_outgoing[0]));
		    outfile__dynamic_array_Syn23_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_X;
	outfile__dynamic_array_Syn23_X.open("results/_dynamic_array_Syn23_X_1854297678", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_X.is_open())
	{
        if (! _dynamic_array_Syn23_X.empty() )
        {
			outfile__dynamic_array_Syn23_X.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_X[0]), _dynamic_array_Syn23_X.size()*sizeof(_dynamic_array_Syn23_X[0]));
		    outfile__dynamic_array_Syn23_X.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_X." << endl;
	}
	ofstream outfile__dynamic_array_Syn23_X_condition;
	outfile__dynamic_array_Syn23_X_condition.open("results/_dynamic_array_Syn23_X_condition_1068277076", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn23_X_condition.is_open())
	{
        if (! _dynamic_array_Syn23_X_condition.empty() )
        {
			outfile__dynamic_array_Syn23_X_condition.write(reinterpret_cast<char*>(&_dynamic_array_Syn23_X_condition[0]), _dynamic_array_Syn23_X_condition.size()*sizeof(_dynamic_array_Syn23_X_condition[0]));
		    outfile__dynamic_array_Syn23_X_condition.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn23_X_condition." << endl;
	}
	ofstream outfile__dynamic_array_Syn33__synaptic_post;
	outfile__dynamic_array_Syn33__synaptic_post.open("results/_dynamic_array_Syn33__synaptic_post_577254464", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33__synaptic_post.is_open())
	{
        if (! _dynamic_array_Syn33__synaptic_post.empty() )
        {
			outfile__dynamic_array_Syn33__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_Syn33__synaptic_post[0]), _dynamic_array_Syn33__synaptic_post.size()*sizeof(_dynamic_array_Syn33__synaptic_post[0]));
		    outfile__dynamic_array_Syn33__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_Syn33__synaptic_pre;
	outfile__dynamic_array_Syn33__synaptic_pre.open("results/_dynamic_array_Syn33__synaptic_pre_1861243049", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33__synaptic_pre.is_open())
	{
        if (! _dynamic_array_Syn33__synaptic_pre.empty() )
        {
			outfile__dynamic_array_Syn33__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_Syn33__synaptic_pre[0]), _dynamic_array_Syn33__synaptic_pre.size()*sizeof(_dynamic_array_Syn33__synaptic_pre[0]));
		    outfile__dynamic_array_Syn33__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_Syn33_delay;
	outfile__dynamic_array_Syn33_delay.open("results/_dynamic_array_Syn33_delay_1159471760", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33_delay.is_open())
	{
        if (! _dynamic_array_Syn33_delay.empty() )
        {
			outfile__dynamic_array_Syn33_delay.write(reinterpret_cast<char*>(&_dynamic_array_Syn33_delay[0]), _dynamic_array_Syn33_delay.size()*sizeof(_dynamic_array_Syn33_delay[0]));
		    outfile__dynamic_array_Syn33_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33_delay." << endl;
	}
	ofstream outfile__dynamic_array_Syn33_N_incoming;
	outfile__dynamic_array_Syn33_N_incoming.open("results/_dynamic_array_Syn33_N_incoming_3341311478", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33_N_incoming.is_open())
	{
        if (! _dynamic_array_Syn33_N_incoming.empty() )
        {
			outfile__dynamic_array_Syn33_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_Syn33_N_incoming[0]), _dynamic_array_Syn33_N_incoming.size()*sizeof(_dynamic_array_Syn33_N_incoming[0]));
		    outfile__dynamic_array_Syn33_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_Syn33_N_outgoing;
	outfile__dynamic_array_Syn33_N_outgoing.open("results/_dynamic_array_Syn33_N_outgoing_3761515820", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33_N_outgoing.is_open())
	{
        if (! _dynamic_array_Syn33_N_outgoing.empty() )
        {
			outfile__dynamic_array_Syn33_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_Syn33_N_outgoing[0]), _dynamic_array_Syn33_N_outgoing.size()*sizeof(_dynamic_array_Syn33_N_outgoing[0]));
		    outfile__dynamic_array_Syn33_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_Syn33_w;
	outfile__dynamic_array_Syn33_w.open("results/_dynamic_array_Syn33_w_2112556146", ios::binary | ios::out);
	if(outfile__dynamic_array_Syn33_w.is_open())
	{
        if (! _dynamic_array_Syn33_w.empty() )
        {
			outfile__dynamic_array_Syn33_w.write(reinterpret_cast<char*>(&_dynamic_array_Syn33_w[0]), _dynamic_array_Syn33_w.size()*sizeof(_dynamic_array_Syn33_w[0]));
		    outfile__dynamic_array_Syn33_w.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_Syn33_w." << endl;
	}

	ofstream outfile__dynamic_array_StateMonitor_L1_v;
	outfile__dynamic_array_StateMonitor_L1_v.open("results/_dynamic_array_StateMonitor_L1_v_1715834261", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L1_v.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L1_v.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L1_v(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L1_v.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L1_v(n, 0)), _dynamic_array_StateMonitor_L1_v.m*sizeof(_dynamic_array_StateMonitor_L1_v(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L1_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L1_v." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L2_g_e;
	outfile__dynamic_array_StateMonitor_L2_g_e.open("results/_dynamic_array_StateMonitor_L2_g_e_2220462755", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L2_g_e.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L2_g_e.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L2_g_e(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L2_g_e.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L2_g_e(n, 0)), _dynamic_array_StateMonitor_L2_g_e.m*sizeof(_dynamic_array_StateMonitor_L2_g_e(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L2_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L2_g_e." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L2_v;
	outfile__dynamic_array_StateMonitor_L2_v.open("results/_dynamic_array_StateMonitor_L2_v_1677931468", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L2_v.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L2_v.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L2_v(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L2_v.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L2_v(n, 0)), _dynamic_array_StateMonitor_L2_v.m*sizeof(_dynamic_array_StateMonitor_L2_v(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L2_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L2_v." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_g_e;
	outfile__dynamic_array_StateMonitor_L3_g_e.open("results/_dynamic_array_StateMonitor_L3_g_e_3107568403", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_g_e.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L3_g_e.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L3_g_e(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L3_g_e.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_g_e(n, 0)), _dynamic_array_StateMonitor_L3_g_e.m*sizeof(_dynamic_array_StateMonitor_L3_g_e(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L3_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_g_e." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_g_i;
	outfile__dynamic_array_StateMonitor_L3_g_i.open("results/_dynamic_array_StateMonitor_L3_g_i_2962224952", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_g_i.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L3_g_i.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L3_g_i(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L3_g_i.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_g_i(n, 0)), _dynamic_array_StateMonitor_L3_g_i.m*sizeof(_dynamic_array_StateMonitor_L3_g_i(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L3_g_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_g_i." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_sum_w;
	outfile__dynamic_array_StateMonitor_L3_sum_w.open("results/_dynamic_array_StateMonitor_L3_sum_w_3678079732", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_sum_w.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L3_sum_w.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L3_sum_w(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L3_sum_w.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_sum_w(n, 0)), _dynamic_array_StateMonitor_L3_sum_w.m*sizeof(_dynamic_array_StateMonitor_L3_sum_w(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L3_sum_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_sum_w." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_v;
	outfile__dynamic_array_StateMonitor_L3_v.open("results/_dynamic_array_StateMonitor_L3_v_1707170299", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_v.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L3_v.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L3_v(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L3_v.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_v(n, 0)), _dynamic_array_StateMonitor_L3_v.m*sizeof(_dynamic_array_StateMonitor_L3_v(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L3_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_v." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_L3_v_th;
	outfile__dynamic_array_StateMonitor_L3_v_th.open("results/_dynamic_array_StateMonitor_L3_v_th_2920312168", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_L3_v_th.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_L3_v_th.n; n++)
        {
            if (! _dynamic_array_StateMonitor_L3_v_th(n).empty())
            {
                outfile__dynamic_array_StateMonitor_L3_v_th.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_L3_v_th(n, 0)), _dynamic_array_StateMonitor_L3_v_th.m*sizeof(_dynamic_array_StateMonitor_L3_v_th(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_L3_v_th.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_L3_v_th." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_Syn23_c;
	outfile__dynamic_array_StateMonitor_Syn23_c.open("results/_dynamic_array_StateMonitor_Syn23_c_3105743036", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_Syn23_c.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_Syn23_c.n; n++)
        {
            if (! _dynamic_array_StateMonitor_Syn23_c(n).empty())
            {
                outfile__dynamic_array_StateMonitor_Syn23_c.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_Syn23_c(n, 0)), _dynamic_array_StateMonitor_Syn23_c.m*sizeof(_dynamic_array_StateMonitor_Syn23_c(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_Syn23_c.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_Syn23_c." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_Syn23_w;
	outfile__dynamic_array_StateMonitor_Syn23_w.open("results/_dynamic_array_StateMonitor_Syn23_w_2747741377", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_Syn23_w.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_Syn23_w.n; n++)
        {
            if (! _dynamic_array_StateMonitor_Syn23_w(n).empty())
            {
                outfile__dynamic_array_StateMonitor_Syn23_w.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_Syn23_w(n, 0)), _dynamic_array_StateMonitor_Syn23_w.m*sizeof(_dynamic_array_StateMonitor_Syn23_w(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_Syn23_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_Syn23_w." << endl;
	}
	ofstream outfile__dynamic_array_StateMonitor_Syn23_X;
	outfile__dynamic_array_StateMonitor_Syn23_X.open("results/_dynamic_array_StateMonitor_Syn23_X_135661976", ios::binary | ios::out);
	if(outfile__dynamic_array_StateMonitor_Syn23_X.is_open())
	{
        for (int n=0; n<_dynamic_array_StateMonitor_Syn23_X.n; n++)
        {
            if (! _dynamic_array_StateMonitor_Syn23_X(n).empty())
            {
                outfile__dynamic_array_StateMonitor_Syn23_X.write(reinterpret_cast<char*>(&_dynamic_array_StateMonitor_Syn23_X(n, 0)), _dynamic_array_StateMonitor_Syn23_X.m*sizeof(_dynamic_array_StateMonitor_Syn23_X(0, 0)));
            }
        }
        outfile__dynamic_array_StateMonitor_Syn23_X.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_StateMonitor_Syn23_X." << endl;
	}
	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
	if(_static_array__array_StateMonitor_L1__indices!=0)
	{
		delete [] _static_array__array_StateMonitor_L1__indices;
		_static_array__array_StateMonitor_L1__indices = 0;
	}
	if(_static_array__array_StateMonitor_L2__indices!=0)
	{
		delete [] _static_array__array_StateMonitor_L2__indices;
		_static_array__array_StateMonitor_L2__indices = 0;
	}
	if(_static_array__array_StateMonitor_L3__indices!=0)
	{
		delete [] _static_array__array_StateMonitor_L3__indices;
		_static_array__array_StateMonitor_L3__indices = 0;
	}
	if(_static_array__array_StateMonitor_Syn23__indices!=0)
	{
		delete [] _static_array__array_StateMonitor_Syn23__indices;
		_static_array__array_StateMonitor_Syn23__indices = 0;
	}
	if(_static_array__array_Syn12_sources!=0)
	{
		delete [] _static_array__array_Syn12_sources;
		_static_array__array_Syn12_sources = 0;
	}
	if(_static_array__array_Syn12_targets!=0)
	{
		delete [] _static_array__array_Syn12_targets;
		_static_array__array_Syn12_targets = 0;
	}
	if(_timedarray_values!=0)
	{
		delete [] _timedarray_values;
		_timedarray_values = 0;
	}
}

