#include "code_objects/StateMonitor_Syn23_codeobject.h"
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



void _run_StateMonitor_Syn23_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const size_t _numN = 1;
double* const _array_Syn23_X = _dynamic_array_Syn23_X.empty()? 0 : &_dynamic_array_Syn23_X[0];
const size_t _num__source_w_Syn23_X = _dynamic_array_Syn23_X.size();
const size_t _num_clock_t = 1;
const size_t _num_indices = 720;
const size_t _num_source_X = _dynamic_array_Syn23_X.size();
double* const _array_Syn23_c = _dynamic_array_Syn23_c.empty()? 0 : &_dynamic_array_Syn23_c[0];
const size_t _num_source_c = _dynamic_array_Syn23_c.size();
double* const _array_StateMonitor_Syn23_t = _dynamic_array_StateMonitor_Syn23_t.empty()? 0 : &_dynamic_array_StateMonitor_Syn23_t[0];
const size_t _numt = _dynamic_array_StateMonitor_Syn23_t.size();
    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_StateMonitor_Syn23_N = _array_StateMonitor_Syn23_N;
    double* __restrict  _ptr_array_Syn23_X = _array_Syn23_X;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    int32_t* __restrict  _ptr_array_StateMonitor_Syn23__indices = _array_StateMonitor_Syn23__indices;
    double* __restrict  _ptr_array_Syn23_c = _array_Syn23_c;
    double* __restrict  _ptr_array_StateMonitor_Syn23_t = _array_StateMonitor_Syn23_t;


    _dynamic_array_StateMonitor_Syn23_t.push_back(_ptr_array_defaultclock_t[0]);

    const size_t _new_size = _dynamic_array_StateMonitor_Syn23_t.size();
    // Resize the dynamic arrays
    _dynamic_array_StateMonitor_Syn23_c.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_Syn23_w.resize(_new_size, _num_indices);
    _dynamic_array_StateMonitor_Syn23_X.resize(_new_size, _num_indices);

    // scalar code
    const size_t _vectorisation_idx = -1;
        


    
    for (int _i = 0; _i < (int)_num_indices; _i++)
    {
        // vector code
        const size_t _idx = _ptr_array_StateMonitor_Syn23__indices[_i];
        const size_t _vectorisation_idx = _idx;
                
        const double __source_w_Syn23_X = _ptr_array_Syn23_X[_idx];
        const double _source_X = _ptr_array_Syn23_X[_idx];
        const double _source_c = _ptr_array_Syn23_c[_idx];
        const double _to_record_X = _source_X;
        const double _source_w = int_(__source_w_Syn23_X > 0.5);
        const double _to_record_w = _source_w;
        const double _to_record_c = _source_c;


        _dynamic_array_StateMonitor_Syn23_c(_new_size-1, _i) = _to_record_c;
        _dynamic_array_StateMonitor_Syn23_w(_new_size-1, _i) = _to_record_w;
        _dynamic_array_StateMonitor_Syn23_X(_new_size-1, _i) = _to_record_X;
    }

    _ptr_array_StateMonitor_Syn23_N[0] = _new_size;


}


