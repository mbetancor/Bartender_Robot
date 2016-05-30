
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <algorithm>
#include <tuple>

#include "server.hpp"

using namespace cv;
using namespace std;

typedef struct {
  char build;
  Mat skin_hist;

  int filter_window;
  int filter_sigma;
  int threshold;
  int canny_thr;
  int length_thr;
  int goodness_thr;
}persistent_t;

persistent_t * data =0;
RNG random_src(12345);


auto show_bars()
{
  createTrackbar("filter_window","::float:: menus",
                 &data->filter_window, 100);
  createTrackbar("filter_sigma","::float:: menus",
                 &data->filter_sigma, 100);
  createTrackbar("threshold","::float:: menus",
                 &data->threshold, 100);
  createTrackbar("canny_thr","::float:: menus",
                 &data->canny_thr, 100);
  createTrackbar("length_thr","::float:: menus",
                 &data->length_thr, 800);
  createTrackbar("goodness_thr","::float:: menus",
                 &data->goodness_thr, 1000);
  

  imshow("::float:: menus",Mat::zeros({400,1},CV_8UC3));
}


class im 
{
 public:
  im(){};
  im(const cv::Mat & m, const std::string & format):mat(m),format(format){}
  im(cv::Mat && m, const std::string & format):mat(move(m)),format(format){}
  
  auto process(Mat && m) -> im&
  {
    mat = std::move(m);
    return *this;
  }

  auto process(im & m) -> im&
  {
    mat = m.mat;
    format = m.format;
    return *this;
  }

  auto operator[](const vector<int> & c) const -> im 
  {
    
    Mat dst(mat.rows,mat.cols,CV_8UC(c.size()));
    vector<char> dst_format;
    vector<int> fromto;
    fromto.reserve(c.size()*2);
    int to = 0;
    for(auto i : c){
      fromto.push_back(i);
      fromto.push_back(to++);
      dst_format.push_back(format[i]);
    }
    mixChannels(&mat,1, &dst,1, fromto.data(),fromto.size()/2);
    return im(dst,{begin(dst_format),end(dst_format)});
  }
    
  auto operator[](const string & c) const -> im
  {
    vector<int> indices(c.size());
    transform(begin(c),end(c),begin(indices),
              [this](auto x){return format.find(x);});
    return operator[](indices);
  }

  template <typename ...Ts>
  auto operator()(const Ts & ...vs) const{
    return tie(((*this)[vs])...);
  }

  auto as_hsv() const -> im
  {
    if(format != "bgr")
      cerr << "Suspicious call to as_hsv() with format " << format << endl; 
    Mat hsv;
    cvtColor( mat, hsv, COLOR_BGR2HSV );
    return im(hsv,"hsv");
  }

  auto as_bgr() const -> im
  {
    if(format != "hsv")
      cerr << "Suspicious call to as_bgr() with format " << format << endl; 
    Mat bgr;
    cvtColor( mat, bgr, COLOR_HSV2BGR );
    return im(bgr,"bgr");
  }

  auto as_lum() const -> im
  {
    Mat l;
    if(format == "hsv")
      return (*this)["v"];
    else if(format == "bgr")
      cvtColor( mat, l, COLOR_BGR2GRAY );
    else{
      cerr << "Cannot cleanly resolve call to as_lum() with format "
           << format << endl;
      return (*this)[0];
    }
    return im(l,"l"); 
  }


  auto mask(const Mat & mask)
  {
    Mat masked;
    mat.copyTo(masked,mask);
    return im(masked,format);
  }
  
  inline operator cv::Mat & (){
    return mat;
  }

  inline operator const cv::Mat & () const{
    return mat;
  }

  
  cv::Mat mat;  
  std::string format; 
};



auto show(const string & title, const Mat & m) -> void
{
  auto final_title = string("::float::ws0::")+title;
  namedWindow(final_title,WINDOW_NORMAL);
  imshow(final_title,m);
}

auto show(const string & title, im & i) -> void
{
  show(title + " " + i.format, i.mat);
}

auto skin_mask(const im & hsv_in,int filter_window, double filter_sigma,
               int threshold) -> im
{
  /// Get Backprojection
  Mat backproj;
  Mat mask;
  Mat filtered_hsv;
  Mat normalized_hist;
  static const float h_range[] = { 0, 179 };
  static const float s_range[] = { 0, 255 };
  static const int channels[] = {0,1};
  static const float * ranges[] = { h_range, s_range };

  normalize(data->skin_hist, normalized_hist, 0, 255, NORM_MINMAX, -1, Mat() );


  calcBackProject(&hsv_in.mat, 1, channels, normalized_hist,
                  backproj, ranges, 1, true );

  GaussianBlur(backproj,backproj,
               {filter_window,filter_window},
               filter_sigma,filter_sigma,BORDER_DEFAULT );
  
  /// Draw the backproj
  show("BackProj", backproj);

  cv::threshold(backproj, mask, threshold, 255, THRESH_BINARY);
  

  return im(mask,"m");
}



auto find_contours(const im & in, im & mask,
                   int thresh = 20, int length_thr = 100,
                   double goodness_thr = 0.5)
{
  static Mat canny_output;
  static Mat drawing;
  static vector<vector<Point> > contours;
  static vector<Vec4i> hierarchy;

  drawing = Mat::zeros( in.mat.rows, in.mat.cols, CV_8UC3 );
  
  //cvtColor(mask.mat, drawing, CV_GRAY2BGR);
  
  Canny(in.mat, canny_output, thresh, thresh*2, 3 );
  findContours(canny_output, contours, hierarchy,
               CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
  
  /// Draw contours
  //Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

  for( int i = 0; i< contours.size(); i++ ){

    if(contours[i].size() > length_thr){
      double goodness = 0;
      for(const auto & point : contours[i]){
        if(mask.mat.at<unsigned char>(point)>100){
          goodness++;
        }
      }
      goodness = goodness/contours[i].size();
      Scalar color = {0,255*goodness,255*(1.-goodness)};//{0, random_src.uniform(0,255), random_src.uniform(128,255) };

      if(goodness > goodness_thr)
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }
  }

  show("Contours", drawing);
}

auto find_contours_bp(const im & in, const im & bp,
                   int thresh = 20)
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  
  Canny(in.mat, canny_output, thresh, thresh*2, 3 );
  findContours(canny_output, contours, hierarchy,
               CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
  
  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  //Mat & drawing = dst.mat;
  for( int i = 0; i< contours.size(); i++ ){
    
    Scalar color = {0,255,0};

    
    if(hierarchy[i][2]<0 and hierarchy[i][3]<0)
      //color = {
 
    
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }

  show("Contours", drawing);
}


auto handle_frame(Mat & m, int number) -> bool
{
  show_bars();
  int    filter_window = ((int(data->filter_window/100.*27.)/2)*2)+3;
  double filter_sigma  = data->filter_sigma/100.*6.;
  double threshold     = data->threshold/100.*20.;
  int    canny_thr     = int(data->canny_thr);
  int    length_thr    = int(data->length_thr);
  double goodness_thr  = double(data->goodness_thr)/1000.0;
  
  static im frame;
  frame = im(m,"bgr").as_hsv();

  static Mat color_mask;
  static Mat filtered;
  
  medianBlur(frame.mat,filtered,filter_window);
  
  static im mask;
  mask = skin_mask(im(filtered,"hsv"),filter_window, filter_sigma, threshold);
  /*
  show("mask", filtered);
  
  auto skin = frame.as_lum().mask(filtered);

  show("skin",skin);
  */

  find_contours(im(filtered,"hsv"), mask, canny_thr, length_thr, goodness_thr);
  
  return true;
}



/* Library interface */
extern "C" EXPORTED void on_load(export_t * e, void ** d){
  cv::FileStorage file;
  e->handle_frame = handle_frame;
  file.open("data.yml", cv::FileStorage::READ);
  if(!*d){
    *d = (void*) new persistent_t;
    data = (persistent_t *) *d;
    file["skin_histogram"] >> data->skin_hist;
    data->build = 0;

    data->filter_window = 32;
    data->filter_sigma  = 50;
    data->threshold     = 50;
    data->canny_thr     = 47;
    data->length_thr    = 100;
    data->goodness_thr  = 1000;
  }
  else
    data = (persistent_t *) *d;
  file.release();
}

extern "C" EXPORTED void on_unload(){
  
}



