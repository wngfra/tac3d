#include "code_objects/Syn23_pre_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
#include "brianlib/stdint_compat.h"
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
        
    template <typename T>
    static inline T _clip(const T value, const double a_min, const double a_max)
    {
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
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



void _run_Syn23_pre_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    double* const _array_Syn23_X = _dynamic_array_Syn23_X.empty()? 0 : &_dynamic_array_Syn23_X[0];
const size_t _numX = _dynamic_array_Syn23_X.size();
double* const _array_Syn23_X_condition = _dynamic_array_Syn23_X_condition.empty()? 0 : &_dynamic_array_Syn23_X_condition[0];
const size_t _numX_condition = _dynamic_array_Syn23_X_condition.size();
const int64_t X_max = 1;
const int64_t X_min = 0;
int32_t* const _array_Syn23__synaptic_pre = _dynamic_array_Syn23__synaptic_pre.empty()? 0 : &_dynamic_array_Syn23__synaptic_pre[0];
const size_t _num_synaptic_pre = _dynamic_array_Syn23__synaptic_pre.size();
const double a = 0.1;
const double b = 0.1;
double* const _array_Syn23_c = _dynamic_array_Syn23_c.empty()? 0 : &_dynamic_array_Syn23_c[0];
const size_t _numc = _dynamic_array_Syn23_c.size();
const size_t _numg_e_post = 36;
const size_t _numsum_w_post = 36;
const int64_t theta_hdown = 4;
const int64_t theta_hup = 12;
const int64_t theta_ldown = 3;
const int64_t theta_lup = 3;
const double theta_v = - 0.04000000000000001;
const size_t _numv_pre = 20;
const double w_e = 3.0000000000000004e-08;
const size_t _numnot_refractory = 20;
int32_t* const _array_Syn23__synaptic_post = _dynamic_array_Syn23__synaptic_post.empty()? 0 : &_dynamic_array_Syn23__synaptic_post[0];
const size_t _num_postsynaptic_idx = _dynamic_array_Syn23__synaptic_post.size();
const size_t _num_presynaptic_idx = _dynamic_array_Syn23__synaptic_pre.size();
    ///// POINTERS ////////////
        
    double* __restrict  _ptr_array_Syn23_X = _array_Syn23_X;
    double* __restrict  _ptr_array_Syn23_X_condition = _array_Syn23_X_condition;
    int32_t* __restrict  _ptr_array_Syn23__synaptic_pre = _array_Syn23__synaptic_pre;
    double* __restrict  _ptr_array_Syn23_c = _array_Syn23_c;
    double* __restrict  _ptr_array_L3_g_e = _array_L3_g_e;
    double* __restrict  _ptr_array_L3_sum_w = _array_L3_sum_w;
    double* __restrict  _ptr_array_L2_v = _array_L2_v;
    char* __restrict  _ptr_array_L2_not_refractory = _array_L2_not_refractory;
    int32_t* __restrict  _ptr_array_Syn23__synaptic_post = _array_Syn23__synaptic_post;



    // This is only needed for the _debugmsg function below

    // scalar code
    const size_t _vectorisation_idx = -1;
        


    
    {
    std::vector<int> *_spiking_synapses = Syn23_pre.peek();
    const int _num_spiking_synapses = _spiking_synapses->size();

    
    {
        for(int _spiking_synapse_idx=0;
            _spiking_synapse_idx<_num_spiking_synapses;
            _spiking_synapse_idx++)
        {
            const size_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
            const size_t _vectorisation_idx = _idx;
                        
            const int32_t _postsynaptic_idx = _ptr_array_Syn23__synaptic_post[_idx];
            const int32_t _presynaptic_idx = _ptr_array_Syn23__synaptic_pre[_idx];
            double X = _ptr_array_Syn23_X[_idx];
            const double c = _ptr_array_Syn23_c[_idx];
            double g_e_post = _ptr_array_L3_g_e[_postsynaptic_idx];
            const double sum_w_post = _ptr_array_L3_sum_w[_postsynaptic_idx];
            const double v_pre = _ptr_array_L2_v[_presynaptic_idx];
            double X_condition;
            g_e_post += 1.0f*(w_e * (int_(sum_w_post >= 1) * X))/(1e-06 + sum_w_post);
            X += (a * ((int_(v_pre > theta_v) * int_(theta_lup < c)) * int_(c < theta_hup))) - (b * ((int_(v_pre <= theta_v) * int_(theta_ldown < c)) * int_(c < theta_hdown)));
            X = _clip(X, X_min, X_max);
            X_condition = ((int_(v_pre > theta_v) * int_(theta_lup < c)) * int_(c < theta_hup)) + ((int_(v_pre <= theta_v) * int_(theta_ldown < c)) * int_(c < theta_hdown));
            _ptr_array_Syn23_X[_idx] = X;
            _ptr_array_Syn23_X_condition[_idx] = X_condition;
            _ptr_array_L3_g_e[_postsynaptic_idx] = g_e_post;

        }
    }
    }

}

void _debugmsg_Syn23_pre_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _dynamic_array_Syn23__synaptic_pre.size() << endl;
}

