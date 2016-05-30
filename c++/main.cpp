// -*- compile-command: "./build.sh" -*-

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <functional>

#include <stdlib.h> // god forgive me
#include <dlfcn.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <curl/curl.h>

#include "server/server.hpp"

using namespace cv;
using namespace std;

/* Colors */
const string BLACK  = "\x1B[30m";
const string RED    = "\x1B[31m";
const string GREEN  = "\x1B[32m";
const string YELLOW = "\x1B[33m";
const string BLUE   = "\x1B[34m";
const string NORMAL = "\x1B[39;49m";


/* Server handling */

static const string  LIB_FILE = "./libhts.so";

export_t exports;
void * server_storage = 0;

static auto load(const string & file) -> bool
{
  static void * dl = nullptr;
  static load_fn_t load_fn = nullptr;
  static unload_fn_t unload_fn = nullptr;
  
  if(unload_fn)
    unload_fn();
  if(dl)
    dlclose(dl);
  
  dl = dlopen(file.c_str(),RTLD_NOW);
  if(!dl) goto fail;
  load_fn  = (load_fn_t) dlsym(dl,"on_load");
  if(!load_fn) goto fail;
  unload_fn = (unload_fn_t) dlsym(dl,"on_unload");
  if(!unload_fn) goto fail;

  load_fn(&exports,&server_storage);

  return true;
 fail:
  cerr << "DL: Error: " << dlerror() << endl;
  return false;
}

static auto is_modified(const string & s, time_t & modified) -> bool
{
  struct stat sb;
  stat(s.c_str(), &sb);
  if(modified != sb.st_mtime){
    modified = sb.st_mtime;
    return true;
  }
  return false;
}

static auto unsafe_handle_frame(Mat & frame) -> bool
{
  static time_t last_modified = 0;
  static int counter = 0;
  if(is_modified(LIB_FILE, last_modified)){
    if(!load(LIB_FILE))
      cerr << "Failed to reload the library" << endl;
  }
  if(exports.handle_frame != nullptr){
    return exports.handle_frame(frame, counter++);
  }
  return true;
}

static auto handle_frame(Mat & frame) -> bool
{
  try {
    return unsafe_handle_frame(frame);
  }
  catch(const cv::Exception & e){
    cerr << RED
         << "OpenCV Exception: " << e.err << endl
         << "In : " << e.file << ":" << e.line << endl
         << "Msg: " << e.msg << endl
         << NORMAL << endl;
  }
  catch(const std::exception & e){
    cerr << RED
         << "STL Exception: " << e.what() << endl
         << NORMAL << endl;
  }
  catch(...){
    cerr << RED
         << "Unknown Exception!"
         << NORMAL << endl;
  }
  exports.handle_frame = nullptr;
  cout << YELLOW << "Discarded handle_frame due to exception." << NORMAL << endl;
  return true;
}



/* CURL frame download */

static size_t http_callback(void *contents,
                            size_t size,
                            size_t nmemb,
                            void *userp)
{
  vector<unsigned char> * bytes = (vector<unsigned char> *) userp;
  size_t real_size = size * nmemb;

  unsigned char * data = (unsigned char*) contents;
  bytes->insert(bytes->end(), data, &data[real_size]);
  
  return real_size;
}

static auto http_get(const string & url, vector<unsigned char> & bytes) -> bool
{
  CURL *curl = nullptr;
  CURLcode res = CURLE_FAILED_INIT;
  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&bytes);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, http_callback);
    
    res = curl_easy_perform(curl);
    if(res != CURLE_OK){
      cerr << "Failed to download frame: " << endl
           << "   " << curl_easy_strerror(res) << endl;
    }
    curl_easy_cleanup(curl);
  }
  
  return res == CURLE_OK;
}

static auto try_decode(vector<unsigned char> bytes, Mat & img) -> bool
{
  try{
    imdecode(bytes,CV_LOAD_IMAGE_COLOR,&img);
  }
  catch(...){
    cerr << "Invalid image!" << endl;
    return false;
  }
  return true;
}

static auto http_get_img(const string & url, Mat & img) -> bool
{
  vector<unsigned char> bytes;
  bool good = false;
  if(http_get(url,bytes)){
    good = try_decode(bytes,img);
  }
  return good;
}

/* Video source */


typedef function<bool(Mat & frame)> frame_source;

// Not very performant, but we don't care here
static bool startswith(const string & s, const string & prefix)
{
  return s.find(prefix) == 0;
}

static frame_source get_source(string src)
{
  if(startswith(src,"http://") or startswith(src,"https://")){
    cout << "Source: CURL (" << src << ")" << endl;
    return [src](Mat & frame){
      return http_get_img(src,frame);
    };
  }
  else if(startswith(src,"video:")){
    int cam_index = 0;
    istringstream ss({src.begin() + 6, src.end()});
    ss >> cam_index;
    cout << "Source: Camera index " << cam_index << endl;
    VideoCapture cap(cam_index);
    if(cap.isOpened()){
      return [cap{move(cap)}](Mat & m) mutable
        {
          return (cap.grab() and cap.retrieve(m));
        };
    }
  }
  else{
    cout << "Source: File (" << src << ")" << endl;
    VideoCapture cap(src);
    if(cap.isOpened()){
      return [cap{move(cap)}](Mat & m) mutable
        {
          Mat f;
          if(cap.grab() and cap.retrieve(f)){
            cv::resize(f, m, m.size(),0,0,INTER_LINEAR);
            return true;
          }
          return false;
        };
    }
  }
  cerr << "Invalid source: " << src << endl;
  return [](Mat &) {return false;};
}

auto print_help(const string & program) -> void
{
  cout << "Usage: " << endl
       << "  " << program << " {-h --help}" << endl
       << "  " << program << " -t <directory>" << endl
       << "  " << program << "<source> [-r]" << endl;
}

auto build_histograms(const string & dir) -> void
{

}


auto feed_server(const string & source, bool loop) -> void
{
  Mat frame =  Mat::zeros( {800,600}, CV_8UC3 );
  bool abort = false;
  bool stall = false;
  bool restart = false;
  do{
    restart = false;
    bool more = true;
    auto s = get_source(source);
    do{
      
      if(stall or (more = s(frame)))
        more = handle_frame(frame);
      
      switch (waitKey(30) & 0xFF){
      case 'q': abort = true; break;
      case ' ': stall ^= true; break;
      case 8  : restart = true; break;
      }
    }while(not restart and more and not abort);
  } while (restart or (loop and not abort));
}



auto main(int argc, char** argv) -> int
{
  if (argc == 1
      or argc > 3
      or string(argv[1]) == "-h"
      or string(argv[1]) == "--help")
  {
    print_help(argv[0]);
    return 0;
  }

  if (string(argv[1]) == "-t"){
    if(argc == 3)
      build_histograms(argv[2]);
    else{
      cerr << "-t expects a directory argument" << endl;
      return 0;
    }
  }

  bool loop = (argc == 3 and string(argv[2]) == "-r");
  feed_server(argv[1], loop);
  return 0;
}
