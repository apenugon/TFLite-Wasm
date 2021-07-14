/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"

namespace tflite {
namespace gpu {
namespace gl {

// EglContext is an RAII wrapper for an EGLContext.
//
// EglContext is moveable but not copyable.
//
// See https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglIntro.xhtml for
// more info.
class EglContext {
 public:
  // Creates an invalid EglContext.
  EglContext()
      : context_(nullptr),
        has_ownership_(false) {}

  EglContext(GLFWwindow* context,
             bool has_ownership)
      : context_(context),
        has_ownership_(has_ownership) {}

  // Move only
  EglContext(EglContext&& other);
  EglContext& operator=(EglContext&& other);
  EglContext(const EglContext&) = delete;
  EglContext& operator=(const EglContext&) = delete;

  ~EglContext() { Invalidate(); }

  GLFWwindow* context() const { return context_; }

  // Make this EglContext the current EGL context on this thread, replacing
  // the existing current.
  absl::Status MakeCurrent();

  absl::Status MakeCurrentSurfaceless() {
    return MakeCurrent();
  }

  // Returns true if this is the currently bound EGL context.
  bool IsCurrent() const;

  // Returns true if this object actually owns corresponding EGL context
  // and manages it's lifetime.
  bool has_ownership() const { return has_ownership_; }

 private:
  void Invalidate();

  GLFWwindow* context_;

  bool has_ownership_;
};

// It uses the EGL_KHR_no_config_context extension to create a no config context
// since most modern hardware supports the extension.
absl::Status CreateConfiglessContext(
                                     EglContext* egl_context);

absl::Status CreateSurfacelessContext(EglContext* egl_context);
/*
absl::Status CreatePBufferContext(EGLDisplay display, EGLContext shared_context,
                                  EglContext* egl_context);
*/
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_
