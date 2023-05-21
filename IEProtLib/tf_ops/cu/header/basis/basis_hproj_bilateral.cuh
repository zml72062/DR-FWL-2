/////////////////////////////////////////////////////////////////////////////
/// \file basis_hproj_bilateral.cuh
///
/// \brief 
///
/// \copyright Copyright (c) 2020 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_HPROJ_BILATERAL_CUH_
#define BASIS_HPROJ_BILATERAL_CUH_

#include "defines.hpp"
#include "basis_interface.cuh"

namespace mccnn{

    /**
     *  Half projection basis function of a bilateral filter.
     *  @paramt D   Number of dimensions.
     *  @paramt K   Number of basis function used.
     *  @paramt U   Number of values per neighbor
     */
    template<int D, int K, int U>
    class HProjBilateralBasis: public BasisInterface<D, K, U>{
 
        public:

            /**
             *  Point correlation.
             */
            enum class ActivationFunction : int { 
                RELU=0,
                LRELU=1,
                ELU=2,
                EXP=3
            };
 
            /**
             *  Constructor.
             *  @param  pAcFunc  Activation function.
             */
            HProjBilateralBasis(ActivationFunction pAcFunc);
 
            /**
             *  Destructor.
             */
            virtual ~HProjBilateralBasis(void);

            /**
             *  Method to compute the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors       Number of neighbors.
             *  @param  pInPtsGPUPtr        Input gpu pointer to the points.
             *  @param  pInSamplesGPUPtr    Input gpu pointer to the samples.
             *  @param  pInInvRadiiGPUPtr   Inverse radius of the receptive field.
             *  @param  pInNeighborsGPUPtr  Input gpu pointer to the neighbor list.
             *  @param  pInPDFsGPUPtr       Input gpu pointer to the pdf values.
             *  @param  pInXNeighValsGPUPtr Input gpu pointer to the per neighbor
             *      information.
             *  @param  pInBasisGPUPtr      Input gpu pointer to the kernel points.
             *  @param  pOutProjGPUPtr      Output pointer to the influences. 
             */
            virtual void compute_basis_proj_pt_coords(
                std::unique_ptr<IGPUDevice>& pDevice,
                const unsigned int pNumNeighbors,       
                const float* pInPtsGPUPtr,
                const float* pInSamplesGPUPtr,
                const float* pInInvRadiiGPUPtr,
                const int* pInNeighborsGPUPtr,
                const float* pInPDFsGPUPtr,
                const float* pInXNeighValsGPUPtr,
                const float* pInBasisGPUPtr,
                float* pOutProjGPUPtr);

            /**
             *  Method to compute the gradients of the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors           Number of neighbors.
             *  @param  pInPtsGPUPtr            Input gpu pointer to the points.
             *  @param  pInSamplesGPUPtr        Input gpu pointer to the samples.
             *  @param  pInInvRadiiGPUPtr       Inverse radius of the receptive field.
             *  @param  pInNeighborsGPUPtr      Input gpu pointer to the neighbor list.
             *  @param  pInPDFsGPUPtr           Input gpu pointer to the pdf values.
             *  @param  pInXNeighValsGPUPtr Input gpu pointer to the per neighbor
             *      information.
             *  @param  pInBasisGPUPtr          Input gpu pointer to the kernel points.
             *  @param  pInGradsGPUPtr          Input gpu pointer to the input gradients.
             *  @param  pOutBasisGradsGPUPtr    Output pointer to the basis gradients. 
             *  @param  pOutPtsGradsGPUPtr      Output pointer to the gradients of
             *      the pointers.
             *  @param  pOutSampleGradsGPUPtr  Output pointer to the gradients of
             *      the samples.
             *  @param  pOutPDFGradsGPUPtr     Output pointer to the gradients of
             *      the pdfs.
             *  @param  pOutXNeighGradsGPUPtr  Output pointer to the gradients of
             *      the pdfs.
             */
            virtual void compute_grads_basis_proj_pt_coords(
                std::unique_ptr<IGPUDevice>& pDevice,
                const unsigned int pNumNeighbors,       
                const float* pInPtsGPUPtr,
                const float* pInSamplesGPUPtr,
                const float* pInInvRadiiGPUPtr,
                const int* pInNeighborsGPUPtr,
                const float* pInPDFsGPUPtr,
                const float* pInXNeighValsGPUPtr,
                const float* pInBasisGPUPtr,
                const float* pInGradsGPUPtr,
                float* pOutBasisGradsGPUPtr,
                float* pOutPtsGradsGPUPtr,
                float* pOutSampleGradsGPUPtr,
                float* pOutPDFGradsGPUPtr,
                float* pOutXNeighGradsGPUPtr);

        private:

            /**Activation function used.*/
            ActivationFunction acFunc_;
    };
}

#endif