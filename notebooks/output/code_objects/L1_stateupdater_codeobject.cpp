#include "code_objects/L1_stateupdater_codeobject.h"
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
        
    static double* _namespace_timedarray_values;
    static inline double _timedarray(const double t, const int i)
    {
        const double epsilon = 0.001000000000000000 / 128;
        if (i < 0 || i >= 400)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/128);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 1502)
            timestep = 1502-1;
        return _namespace_timedarray_values[timestep*400 + i];
    }
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



void _run_L1_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double C_mem = 2e-10;
const int64_t N = 400;
const double V_res = - 0.06;
const size_t _numdt = 1;
const double g_l = 1e-08;
const size_t _numi = 400;
const size_t _numlastspike = 400;
const size_t _numnot_refractory = 400;
const size_t _numt = 1;
const double tau_r = 0.005;
const size_t _numv = 400;
    ///// POINTERS ////////////
        
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    int32_t* __restrict  _ptr_array_L1_i = _array_L1_i;
    double* __restrict  _ptr_array_L1_lastspike = _array_L1_lastspike;
    char* __restrict  _ptr_array_L1_not_refractory = _array_L1_not_refractory;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_L1_v = _array_L1_v;
    _namespace_timedarray_values = _timedarray_values;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double t = _ptr_array_defaultclock_t[0];
    const int64_t _lio_1 = _timestep(tau_r, dt);
    const double _lio_2 = 1.0f*dt/C_mem;


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        const int32_t i = _ptr_array_L1_i[_idx];
        const double lastspike = _ptr_array_L1_lastspike[_idx];
        char not_refractory = _ptr_array_L1_not_refractory[_idx];
        double v = _ptr_array_L1_v[_idx];
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        double _v;
        if(!not_refractory)
            _v = v;
        else 
            _v = v + (_lio_2 * ((g_l * (V_res - v)) + _timedarray(t, i)));
        if(not_refractory)
            v = _v;
        _ptr_array_L1_not_refractory[_idx] = not_refractory;
        _ptr_array_L1_v[_idx] = v;

    }

}


