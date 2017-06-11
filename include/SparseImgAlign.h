#ifndef SPARSE_IMG_ALIGN_H
#define SPARSE_IMG_ALIGN_H

#include <vikit/vision.h>
#include <vikit/nlls_solver.h>
#include <vikit/performance_monitor.h>
#include "global.h"

#include <Frame.h>
#include <MapPoint.h>

namespace vk {
class AbstractCamera;
}

namespace ORB_SLAM2 {

class Frame;
class MapPoint;

inline void jacobian_xyz2uv(
  const Vector3d& xyz_in_f,
  Matrix<double,2,6>& J);

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlign : public vk::NLLSSolver<6, Sophus::SE3>
{
  static const int patch_halfsize_ = 2;
  static const int patch_size_ = 2*patch_halfsize_;
  static const int patch_area_ = patch_size_*patch_size_;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<cv::Mat> ImgPyr;

  cv::Mat resimg_;

  SparseImgAlign(
      int max_level,
      int min_level,
      int n_iter,
      Method method,
      bool display,
      bool verbose);

  size_t run(
      FramePtr ref_frame,
      FramePtr cur_frame);

  /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
  /// at the converged state.
  Matrix<double, 6, 6> getFisherInformation();

protected:
  FramePtr ref_frame_;            //!< reference frame, has depth for gradient pixels.
  FramePtr cur_frame_;            //!< only the image is known!
  int level_;                     //!< current pyramid level on which the optimization runs.
  bool display_;                  //!< display residual image.
  int max_level_;                 //!< coarsest pyramid level for the alignment.
  int min_level_;                 //!< finest pyramid level for the alignment.

  // cache:
  Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_cache_;
  bool have_ref_patch_cache_;
  cv::Mat ref_patch_cache_;
  std::vector<bool> visible_fts_;

  // image pyramid
  ImgPyr vRefImgPyr;
  ImgPyr vCurImgPyr;

  void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

  void precomputeReferencePatches();
  virtual double computeResiduals(const Sophus::SE3& model,
                                  bool linearize_system,
                                  bool compute_weight_scale = false);
  virtual int solve();
  virtual void update (const ModelType& old_model,
                       ModelType& new_model);
  virtual void startIteration();
  virtual void finishIteration();

};

} // namespace ORN_SLAM2

#endif // SPARSE_IMG_ALIGN_H