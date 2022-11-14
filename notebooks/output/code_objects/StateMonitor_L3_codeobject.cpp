#include "code_objects/StateMonitor_L3_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<int,long double> { typedef long double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long,long double> { typedef long double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<long long,long double> { typedef long double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int> { typedef long double type; };
    template < > struct _higher_type<long double,long> { typedef long double type; };
    template < > struct _higher_type<long double,long long> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_StateMonitor_L3_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const size_t _numN = 1;
const size_t _num_clock_t = 1;
const size_t _num_indices = 36;
const size_t _num_source_g_e = 36;
const size_t _num_source_g_i = 36;
const size_t _num_source_sum_w = 36;
const size_t _num_source_v = 36;
const size_t _num_source_v_th = 36;
double* const _array_StateMonitor_L3_t = _dynamic_array_StateMonitor_L3_t.empty()? 0 : &_dynamic_array_StateMonitor_L3_t[0];
const size_t _numt = _dynamic_array_StateMonitor_L3_t.size();
const size_t _numnot_refractory = 36;
    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_StateMonitor_L3_N = _array_StateMonitor_L3_N;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    int32_t* __restrict  _ptr_array_StateMonitor_L3__indices = _array_StateMonitor_L3__indices;
    double* __restrict  _ptr_array_L3_g_e = _array_L3_g_e;
    double* __restrict  _ptr_array_L3_g_i = _array_L3_g_i;
    double* __restrict  _ptr_array_L3_sum_w = _array_L3_sum_w;
    double* __restrict  _ptr_array_L3_v = _array_L3_v;
    double* __restrict  _ptr_array_L3_v_th = _array_L3_v_th;
    double* __restrict  _ptr_array_StateMonitor_L3_t = _array_StateMonitor_L3_t;
    char* __restrict  _ptr_array_L3_not_refractory = _array_L3_not_refractory;


    _dynamic_array_StateMonitor_L3_t.push_back(_ptr_array_defaultclock_t[0]);

    const size_t _new_size = _dynamic_array_StateMonitor_L3_t.size();
    // Resize the dynamic arrays
    _dynamic_array_StateMonitor_L3_g_e.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_L3_g_i.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_L3_sum_w.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_L3_v.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_L3_v_th.resize(_new_size, _num_indices);

    // scalar code
    const size_t _vectorisation_idx = -1;
        


    
    for (int _i = 0; _i < (int)_num_indices; _i++)
    {
        // vector code
        const size_t _idx = _ptr_array_StateMonitor_L3__indices[_i];
        const size_t _vectorisation_idx = _idx;
                
        const double _source_g_e = _ptr_array_L3_g_e[_idx];
        const double _source_g_i = _ptr_array_L3_g_i[_idx];
        const double _source_sum_w = _ptr_array_L3_sum_w[_idx];
        const double _source_v = _ptr_array_L3_v[_idx];
        const double _source_v_th = _ptr_array_L3_v_th[_idx];
        const double _to_record_v = _source_v;
        const double _to_record_v_th = _source_v_th;
        const double _to_record_g_e = _source_g_e;
        const double _to_record_g_i = _source_g_i;
        const double _to_record_sum_w = _source_sum_w;


        _dynamic_array_StateMonitor_L3_g_e(_new_size-1, _i) = _to_record_g_e;
        _dynamic_array_StateMonitor_L3_g_i(_new_size-1, _i) = _to_record_g_i;
        _dynamic_array_StateMonitor_L3_sum_w(_new_size-1, _i) = _to_record_sum_w;
        _dynamic_array_StateMonitor_L3_v(_new_size-1, _i) = _to_record_v;
        _dynamic_array_StateMonitor_L3_v_th(_new_size-1, _i) = _to_record_v_th;
    }

    _ptr_array_StateMonitor_L3_N[0] = _new_size;


}


