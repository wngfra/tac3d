#include "code_objects/L3_stateupdater_codeobject.h"
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
        
    static inline int64_t _timestep(double t, double dt)
    {
        return (int64_t)((t + 1e-3*dt)/dt); 
    }
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



void _run_L3_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double C_mem = 2e-10;
const int64_t N = 36;
const double V_ir = - 0.08;
const double V_res = - 0.06;
const double V_theta = - 0.05;
const size_t _numdt = 1;
const size_t _numg_e = 36;
const size_t _numg_i = 36;
const double g_l = 1e-08;
const size_t _numlastspike = 36;
const size_t _numnot_refractory = 36;
const size_t _numt = 1;
const double tau_e = 0.005;
const double tau_i = 0.005;
const double tau_r = 0.005;
const double tau_theta = 0.005;
const size_t _numv = 36;
const size_t _numv_th = 36;
    ///// POINTERS ////////////
        
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_L3_g_e = _array_L3_g_e;
    double* __restrict  _ptr_array_L3_g_i = _array_L3_g_i;
    double* __restrict  _ptr_array_L3_lastspike = _array_L3_lastspike;
    char* __restrict  _ptr_array_L3_not_refractory = _array_L3_not_refractory;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_L3_v = _array_L3_v;
    double* __restrict  _ptr_array_L3_v_th = _array_L3_v_th;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double t = _ptr_array_defaultclock_t[0];
    const int64_t _lio_1 = _timestep(tau_r, dt);
    const double _lio_2 = 1.0f*(- dt)/tau_e;
    const double _lio_3 = 1.0f*(- dt)/tau_i;
    const double _lio_4 = 1.0f*dt/C_mem;
    const double _lio_5 = 1.0f*dt/tau_theta;


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        double g_e = _ptr_array_L3_g_e[_idx];
        double g_i = _ptr_array_L3_g_i[_idx];
        const double lastspike = _ptr_array_L3_lastspike[_idx];
        char not_refractory = _ptr_array_L3_not_refractory[_idx];
        double v = _ptr_array_L3_v[_idx];
        double v_th = _ptr_array_L3_v_th[_idx];
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const double _g_e = (_lio_2 * g_e) + g_e;
        const double _g_i = (_lio_3 * g_i) + g_i;
        double _v;
        if(!not_refractory)
            _v = v;
        else 
            _v = v + (_lio_4 * ((((- g_e) * v) + (g_i * (V_ir - v))) + (g_l * (V_res - v))));
        const double _v_th = (_lio_5 * (V_theta - v_th)) + v_th;
        g_e = _g_e;
        g_i = _g_i;
        if(not_refractory)
            v = _v;
        v_th = _v_th;
        _ptr_array_L3_g_e[_idx] = g_e;
        _ptr_array_L3_g_i[_idx] = g_i;
        _ptr_array_L3_not_refractory[_idx] = not_refractory;
        _ptr_array_L3_v[_idx] = v;
        _ptr_array_L3_v_th[_idx] = v_th;

    }

}


