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

#include "tensorflow/lite/delegates/gpu/gl/egl_context.h"

#include <cstring>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {
/*
absl::Status GetConfig(EGLDisplay display, const EGLint* attributes,
                       EGLConfig* config) {
  EGLint config_count;
  bool chosen = eglChooseConfig(display, attributes, config, 1, &config_count);
  RETURN_IF_ERROR(GetEglError());
  if (!chosen || config_count == 0) {
    return absl::InternalError("No EGL error, but eglChooseConfig failed.");
  }
  return absl::OkStatus();
}*/

absl::Status CreateContext(
                           EglContext* egl_context) {
#ifdef _DEBUG  // Add debugging bit
#endif
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  GLFWwindow* context = glfwCreateWindow(32, 32, "", NULL, NULL);
  //EGLContext context =
  //    eglCreateContext(display, config, shared_context, attributes);
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (context == nullptr) {
    return absl::InternalError("No EGL error, but eglCreateContext failed.");
  }
  *egl_context = EglContext(context, true);
  return absl::OkStatus();
}
/*
bool HasExtension(EGLDisplay display, const char* name) {
  return std::strstr(eglQueryString(display, EGL_EXTENSIONS), name);
}*/

}  // namespace

void EglContext::Invalidate() {
  if (context_ != nullptr) {
    if (has_ownership_) {
      glfwMakeContextCurrent(context_);
      glfwDestroyWindow(context_);
    }
    context_ = nullptr;
  }
  has_ownership_ = false;
}

EglContext::EglContext(EglContext&& other)
    : context_(other.context_),
      has_ownership_(other.has_ownership_) {
  other.context_ = nullptr;
  other.has_ownership_ = false;
}

EglContext& EglContext::operator=(EglContext&& other) {
  if (this != &other) {
    Invalidate();
    using std::swap;
    swap(context_, other.context_);
    swap(has_ownership_, other.has_ownership_);
  }
  return *this;
}

absl::Status EglContext::MakeCurrent() {
  glfwMakeContextCurrent(context_);
  RETURN_IF_ERROR(GetOpenGlErrors());
  //if (!is_made_current) {
  //  return absl::InternalError("No EGL error, but eglMakeCurrent failed.");
  //}
  return absl::OkStatus();
}

bool EglContext::IsCurrent() const {
  return context_ == glfwGetCurrentContext();
}

absl::Status CreateConfiglessContext(
                                     EglContext* egl_context) {
  //if (!HasExtension(display, "EGL_KHR_no_config_context")) {
  //  return absl::UnavailableError("EGL_KHR_no_config_context not supported");
  //}
  return CreateContext(egl_context);
}

absl::Status CreateSurfacelessContext(
                                      EglContext* egl_context) {
  //if (!HasExtension(display, "EGL_KHR_create_context")) {
  //  return absl::UnavailableError("EGL_KHR_create_context not supported");
  //}
  //if (!HasExtension(display, "EGL_KHR_surfaceless_context")) {
  //  return absl::UnavailableError("EGL_KHR_surfaceless_context not supported");
  //}
  return CreateContext(egl_context);
}
/*
absl::Status CreatePBufferContext(EGLDisplay display, EGLContext shared_context,
                                  EglContext* egl_context) {
  const EGLint attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,     EGL_BIND_TO_TEXTURE_RGB,
      EGL_TRUE,         EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_NONE};
  EGLConfig config;
  RETURN_IF_ERROR(GetConfig(display, attributes, &config));
  return CreateContext(display, shared_context, config, egl_context);
}
*/
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
