#include "code_objects/Syn23_stateupdater_codeobject.h"
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



void _run_Syn23_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double Hz = 1.0;
const int64_t J_C = 1;
const size_t _numN = 1;
double* const _array_Syn23_X = _dynamic_array_Syn23_X.empty()? 0 : &_dynamic_array_Syn23_X[0];
const size_t _numX = _dynamic_array_Syn23_X.size();
double* const _array_Syn23_X_condition = _dynamic_array_Syn23_X_condition.empty()? 0 : &_dynamic_array_Syn23_X_condition[0];
const size_t _numX_condition = _dynamic_array_Syn23_X_condition.size();
const int64_t X_max = 1;
const int64_t X_min = 0;
const double alpha = 3.5;
const double beta = 3.5;
double* const _array_Syn23_c = _dynamic_array_Syn23_c.empty()? 0 : &_dynamic_array_Syn23_c[0];
const size_t _numc = _dynamic_array_Syn23_c.size();
double* const _array_Syn23_count = _dynamic_array_Syn23_count.empty()? 0 : &_dynamic_array_Syn23_count[0];
const size_t _numcount = _dynamic_array_Syn23_count.size();
const size_t _numdt = 1;
const double tau_c = 0.06;
const double theta_X = 0.5;
    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_Syn23_N = _array_Syn23_N;
    double* __restrict  _ptr_array_Syn23_X = _array_Syn23_X;
    double* __restrict  _ptr_array_Syn23_X_condition = _array_Syn23_X_condition;
    double* __restrict  _ptr_array_Syn23_c = _array_Syn23_c;
    double* __restrict  _ptr_array_Syn23_count = _array_Syn23_count;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double _lio_1 = Hz * J_C;
    const double _lio_2 = 1.0f*1.0/tau_c;


    const int _N = _array_Syn23_N[0];
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        double X = _ptr_array_Syn23_X[_idx];
        const double X_condition = _ptr_array_Syn23_X_condition[_idx];
        double c = _ptr_array_Syn23_c[_idx];
        const double count = _ptr_array_Syn23_count[_idx];
        const double _X = X + (dt * ((1.0 - X_condition) * ((alpha * (int_(X > theta_X) * int_(X < X_max))) - (beta * (int_(X <= theta_X) * int_(X > X_min))))));
        const double _c = c + (dt * ((_lio_1 * count) - (_lio_2 * c)));
        X = _X;
        c = _c;
        _ptr_array_Syn23_X[_idx] = X;
        _ptr_array_Syn23_c[_idx] = c;

    }

}


