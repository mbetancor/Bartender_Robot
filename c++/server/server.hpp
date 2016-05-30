#pragma once

#define EXPORTED __attribute__((__visibility__("default")))

typedef struct 
{
  bool (* handle_frame) (cv::Mat & frame, int number);
} export_t;


typedef void (*unload_fn_t) ();
typedef void (*load_fn_t) (export_t*, void **);
