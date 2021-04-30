//
// Created by wangweihan on 8/2/20.
//

#include "SparsePipeline.h"
#include "Converter.h"
#include<iostream>
#include <Eigen/StdVector>



namespace AVP
{

SparsePipline::SparsePipline(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];
    mb = fSettings["Camera.baseline"];
    
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;


    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    cout<<"- stereo baseline: "<<mb<<endl;
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
}

void SparsePipline::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    cv::Mat mImGray = imRectLeft;

    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    //Step 1: Extract ORB keypoints from the left image and find the corresponding  keypoint from the right image
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mK,mDistCoef,mbf,mThDepth);

    cout << "No t. "<< setw(20)<<setprecision(9)<<setiosflags(ios::fixed)<<timestamp<<endl;

//     vector<Point<float>> vPoints;
//     vector<float> vDepth;

//     //SLAM method
//     Eigen::Matrix4d currentPose = Track();
//     mRwc = currentPose.block<3,3>(0,0);
//     mtwc = currentPose.block<3,1>(0,3);

//     //Step 2: Apply Delaunay triangulation on the keypoints
//     DelaunayTriangulation(vPoints, vDepth);

//     // Step 3: Select as the active keypoint the one with smallest expected reconstruction error
//     int activekpIndex = SelectActiveKeyPoint();
//     cout<<activekpIndex<<endl;
//     //Step 4: Compute the NBV for the active keypoint

//     //Step 4.1 Determine the next best target location in the relative frame
//     //Eigen::Vector3d NextBestTargetLocation(Eigen::Vector3d &p_k, Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior);


//     vector<pair<float, int> > vDepthIdx;
//     vDepthIdx.reserve(mCurrentFrame.N);
//     for(int iL = 0; iL < mCurrentFrame.N; iL++)
//     {
//         if(mCurrentFrame.mvDepth[iL]>0)
//             vDepthIdx.push_back({mCurrentFrame.mvDepth[iL], iL});
//     }
//     sort(vDepthIdx.begin(), vDepthIdx.end(),[](const pair<float, int>  &a, const pair<float, int>  &b){return a.first< b.first;});

//     cv::Mat im, im2;
//     imRectLeft.copyTo(im);
//     imRectRight.copyTo(im2);
//     cvtColor(im, im, CV_GRAY2RGB);
//     cvtColor(im2, im2, CV_GRAY2RGB);

//     const float r = 5;


//     //end
//     for(int i = 0; i < vDepthIdx.size(); i++)
//     {
//         if(vDepthIdx[i].second == activekpIndex)
//         {
//             cv::Point2f pt1,pt2; //float
//             pt1.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-r;
//             pt1.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
//             pt2.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x+r;
//             pt2.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;

//             cout<<"active keypoint depth (other color): "<< vDepthIdx[i].first<<endl;
//             cv::rectangle(im,pt1,pt2,cv::Scalar(120,120,120)); // green close
//             cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(120,120,120),-1);

//         }

//         if (i==0 || i == vDepthIdx.size()/2 || i == vDepthIdx.size()-1){
//             cv::Point2f pt1,pt2;
//             pt1.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-r;
//             pt1.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
//             pt2.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x+r;
//             pt2.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;

//             //cv::Point2f pt1R,pt2R;
//             double dispar = mCurrentFrame.mbf/vDepthIdx[i].first;
//             //pt1R.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar-r;
//             //pt1R.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
//             //pt2R.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar+r;
//             //pt2R.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;
//             cv::Point2f cur(cv::Point2f(mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar, mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y));

//             if (i == 0){
//                 //cout<<"close dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
//                 cout<<"close depth (green): "<< vDepthIdx[i].first<<endl;
//                 cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0)); // green close
//                 cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(0,255,0),-1);

//                 //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(0,255,0)); // green close
//                 //cv::circle(im2,cur,2,cv::Scalar(0,255,0),-1);

//             } else if (i == vDepthIdx.size()/2){
//                 //cout<<"between dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
//                 cout<<"between depth (blue): "<< vDepthIdx[i].first<<endl;
//                 cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));   // blue
//                 cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(255,0,0),-1);

//                 //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(255,0,0));   // blue
//                 //cv::circle(im2,cur,2,cv::Scalar(255,0,0),-1);
//             } else {
//                 //cout<<"far dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
//                 cout<<"far depth (red): "<< vDepthIdx[i].first<<endl;
//                 cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,255));   // red far
//                 cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(0,0,255),-1);

//                 //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(0,0,255));   // red far
//                 //cv::circle(im2,cur,2,cv::Scalar(0,0,255),-1);

//             }


//         }
//     }

// //    cv::Point2f pt1,pt2;
// //
// //    pt1.x=mCurrentFrame.mvKeysUn[activekpIndex].pt.x-r;
// //    pt1.y=mCurrentFrame.mvKeysUn[activekpIndex].pt.y-r;
// //    pt2.x=mCurrentFrame.mvKeysUn[activekpIndex].pt.x+r;
// //    pt2.y=mCurrentFrame.mvKeysUn[activekpIndex].pt.y+r;
// //    cout<<"activekpIndex: "<< activekpIndex<<endl;
// //    cout<<"Depth of this kp: "<<mCurrentFrame.mvDepth[activekpIndex]<<endl;
// //    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
// //    cv::circle(im,mCurrentFrame.mvKeysUn[activekpIndex].pt,2,cv::Scalar(0,255,0),-1);
//     cout<<"activekp index: "<<activekpIndex<<", close kp index: "<<vDepthIdx[0].second<<endl;
//     //cv::imshow("AVP: Current Frame ", im);
    //cv::imshow("AVP: Current Right Frame", im2);
    //cv::waitKey(0);

}

//track keypoints from last frame and determine if keypoints in current frame is old or not
vector<int> SparsePipline::TrackKeyPoints(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool mbCheckOrientation)
{   
    cout<<"Start to track keypoint in current frame"<<endl;
    // Rotation Histogram (to check rotation consistency)
    const int HISTO_LENGTH = 30;
    const int TH_HIGH = 100;

    const float fx = mK.at<float>(0,0);
    const float fy = mK.at<float>(1,1);
    const float cx = mK.at<float>(0,2);
    const float cy = mK.at<float>(1,2);
    
    const float invfx = 1.0f/fx;
    const float invfy = 1.0f/fy;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    

    const float factor = HISTO_LENGTH/360.0f;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    const bool bForward = tlc.at<float>(2)>mb;
    const bool bBackward = -tlc.at<float>(2)>mb;

    
    for(int i=0; i<LastFrame.N; i++)  //LastFrame.N
    {
       if(LastFrame.mvuRight[i] >= 0)  //not occluded
       {
            const float u_lastFrame = LastFrame.mvKeys[i].pt.x;
            const float v_lastFrame = LastFrame.mvKeys[i].pt.y;
            const float z = LastFrame.mvDepth[i];
            const float x = (u_lastFrame - cx)*z*invfx;
            const float y =(v_lastFrame - cy)*z*invfy;

            cv::Mat x3Dc_lastFrame = (cv::Mat_<float>(3,1)<<x, y, z);

            cv::Mat x3Dw = LastFrame.mRwc * x3Dc_lastFrame + LastFrame.mtwc;

            cv::Mat x3Dc_currentFrame = Rcw*x3Dw+tcw;

            const float xc = x3Dc_currentFrame.at<float>(0);
            const float yc = x3Dc_currentFrame.at<float>(1);
            const float invzc = 1.0/x3Dc_currentFrame.at<float>(2);

            if(invzc<0)
                continue;
            
            
            float u_currentFrame = fx*xc*invzc+cx;
            float v_currentFrame = fy*yc*invzc+cy;

            if(u_currentFrame<CurrentFrame.mnMinX || u_currentFrame>CurrentFrame.mnMaxX)
                continue;
            if(v_currentFrame<CurrentFrame.mnMinY || v_currentFrame>CurrentFrame.mnMaxY)
                continue;

            int nLastOctave = LastFrame.mvKeys[i].octave;

            float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

            vector<size_t> vIndices2;

            if(bForward)
                vIndices2 = CurrentFrame.GetFeaturesInArea(u_currentFrame, v_currentFrame, radius, nLastOctave);
            else if(bBackward)
                vIndices2 = CurrentFrame.GetFeaturesInArea(u_currentFrame, v_currentFrame, radius, 0, nLastOctave);
            else 
                vIndices2 = CurrentFrame.GetFeaturesInArea(u_currentFrame, v_currentFrame, radius, nLastOctave-1, nLastOctave+1);
            
            if(vIndices2.empty())
                continue;

            const cv::Mat dlastFrame = LastFrame.mDescriptors.row(i);

            int bestDist = 256;
            int bestIdx2 = -1;

            for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
            {
                const size_t i2 = *vit;
                if(CurrentFrame.mvmatchedNewKeypointsIndex[i2]>=0) //this keypoint in current frame has been matched.
                        continue;

                if(CurrentFrame.mvuRight[i2]>0)
                {
                    const float ur = u_currentFrame - CurrentFrame.mbf*invzc;
                    const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                    if(er>radius)
                        continue;
                }

                const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                const int dist = DescriptorDistance(dlastFrame,d);

                if(dist<bestDist)
                {
                    bestDist=dist;
                    bestIdx2=i2;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                if(CurrentFrame.mvuRight[bestIdx2]>=0)
                {
                    CurrentFrame.mvmatchedNewKeypointsIndex[bestIdx2] = i;
                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeys[i].angle-CurrentFrame.mvKeys[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
                   
            }

       } 
       
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvmatchedNewKeypointsIndex[rotHist[i][j]]=-1;
                    
                }
            }
        }
    }

    return CurrentFrame.mvmatchedNewKeypointsIndex;
}

void SparsePipline::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

void SparsePipline::DelaunayTriangulation(vector<Point<float>> &vPoints, vector<float> &vDepth)
{

    int nodeId = 0;
    for(int iL = 0; iL < mCurrentFrame.N; iL++)
    {
        if(mCurrentFrame.mvDepth[iL]!=-1)
        {
            const cv::KeyPoint &kpL = mCurrentFrame.mvKeysUn[iL];
            const float &uL = kpL.pt.x;
            const float &vL = kpL.pt.y;
            vPoints.push_back({uL, vL, nodeId++});
            vDepth.push_back(mCurrentFrame.mvDepth[iL]);
        }
    }

    const Delaunay<float>  triangulation = triangulate(vPoints);

    WritePTCloud(vPoints, vDepth, triangulation);
}

Eigen::Matrix4d SparsePipline::Track()
{
    mState = NOT_INITIALIZED;

    if(mState==NOT_INITIALIZED)
    {
        StereoInitialization();
    }
}

void SparsePipline::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        mState=OK;
    }
}

// int SparsePipline::SelectActiveKeyPoint()
// {
    
//     int N = mCurrentFrame.mvKeyPointsPostCovariance.size();
    
//     float best_trace = 1.0*INT_MAX;
//     int activeKeyPointIndex = -1;

//     for(size_t i = 0; i < N ; ++i)
//     {
//         if(mCurrentFrame.mvuRight[i]<0) //this keypoint is occluded.
//             continue;
        
//         float cur_trace = mCurrentFrame.mvKeyPointsPostCovariance[i].trace();

//         if(cur_trace < best_trace)
//         {
//             best_trace = cur_trace;
//             activeKeyPointIndex = i;
//         }

//     }

//     return activeKeyPointIndex;
// }

//Gradient of objective function respect to target position using cyclopean coordinate
Eigen::Vector3d SparsePipline::GradientForTarget(const Eigen::Vector3d &input, const float &depth, const Eigen::Matrix3d &U_prior, const Eigen::Matrix3d &Rwc)
{
    //cout<<"Enter Gradient Test function"<<endl;
    //transfer target's pixel coordinate to cyclopean coordinate
    float fx = mK.at<float>(0,0);
    float fy = mK.at<float>(1,1);
    float cx = mK.at<float>(0,2);
    float cy = mK.at<float>(1,2);

    Eigen::Vector3d p_k_1; //location at time k-1
    const float &xL = input[0];
    const float &xR = input[1];
    const float &y = input[2];  //pixel y
    
    double front_p = mb/(xL-xR);
    
    //active kepoint location in relative frame at time k-1 in cyclopean coordinate
    p_k_1 << front_p*0.5*(xL + xR), front_p*y, front_p*fx;

  
    Eigen::Matrix3d J = MakeJacobian(input);
    
    int level = 8.0*depth/10.0;
    
    if(level >= 8)
        level = 7;

    Eigen::Matrix3d Q = MakeQ(level);

    //partial differential of U_posterior at time k respect to [pk]v  dU_post/d[pk]v, v = x,y,z //target location
    Eigen::Vector3d dUpost_dpv;  //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]

    Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose();
 
    //cout<<"Uobs: "<<Uobs<<endl;
    //cout<<"U_prior: "<<U_prior<<endl;
    Eigen::Matrix3d U_post = (U_prior.inverse()+Uobs.inverse()).inverse();
   
    Eigen::Matrix3d U_post_2 = U_post*U_post;
    
    Eigen::Matrix3d front_Uobs = Uobs.inverse()*U_post_2*Uobs.inverse(); //front equation (16) and (10) in 2d paper

 
    //v = x
    //Calculate related to J, px
    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpx;
    //equation (19) in paper
    Eigen::Matrix3d dJ_dxL, dJ_dxR, dJ_dy;  //size: 3*3
    double dxL_dpx, dxR_dpx, dy_dpx;

    //common variable
    //(xL-xR)^3
    double xL_xR_3 = (xL-xR)*(xL-xR)*(xL-xR);
    //(xL-xR)^2
    double xL_xR_2 = (xL-xR)*(xL-xR);

    //dJ_dxL 
    dJ_dxL << 2*mb*xR/xL_xR_3, mb*(-xL-xR)/xL_xR_3, 0,
            2*mb*y/xL_xR_3, -2*mb*y/xL_xR_3, -mb/xL_xR_2,
            2*mb*fx/xL_xR_3, -2*mb*fx/xL_xR_3, 0;
    

    //dJ_dxR 
    dJ_dxR << -mb*(xR+xL)/xL_xR_3, 2*mb*xL/xL_xR_3, 0,
            -2*mb*y/xL_xR_3, 2*mb*y/xL_xR_3, mb/xL_xR_2,
            -2*mb*fx/xL_xR_3, 2*mb*fx/xL_xR_3, 0;
    

    //dJ_dy
    dJ_dy << 0, 0, 0,
            -mb/xL_xR_2, mb/xL_xR_2, 0,
            0, 0, 0;
    

    //Calculate related to J^T, px
    //dJT_px = dJT_dxL * dxL_dpx + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpx;
    Eigen::Matrix3d dJT_dxL, dJT_dxR, dJT_dy;  //size: 3*3

    //dJT_dxL
    dJT_dxL << 2*mb*xR/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*fx/xL_xR_3,
            mb*(-xL-xR)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*fx/xL_xR_3,
            0, -mb/xL_xR_2, 0;

    dJT_dxL = dJ_dxL.transpose();

    //dJT_dxR
    dJT_dxR << -mb*(xR+xL)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*fx/xL_xR_3,
            2*mb*xL/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*fx/xL_xR_3,
            0, mb/xL_xR_2, 0;
    dJT_dxR = dJ_dxR.transpose(); 
    //dJT_dy
    dJT_dy << 0, -mb/xL_xR_2, 0,
            0, mb/xL_xR_2, 0,
            0, 0, 0;
    dJT_dy = dJ_dy.transpose(); 
    
    //v = x
    //dxL_dpx
    dxL_dpx  = fx/p_k_1[2];

    //dxR_dpx
    dxR_dpx  = fx/p_k_1[2];

    //dy_dpx
    dy_dpx  = 0;

    dJ_dpx = dJ_dxL*dxL_dpx + dJ_dxR*dxR_dpx + dJ_dy*dy_dpx;
    dJT_dpx = dJT_dxL*dxL_dpx + dJT_dxR*dxR_dpx + dJT_dy*dy_dpx;

    Eigen::Matrix3d dUobs_dpx = Rwc*(dJ_dpx*Q*J.transpose() +J*Q*dJT_dpx)*Rwc.transpose();
    
    //cout<<"dUobs_dpx: "<<dUobs_dpx<<endl;
    //v = y
    //Calculate related to J, py
    //equation (19) in paper
    double dxL_dpy, dxR_dpy, dy_dpy;

    //dxL_dpy
    dxL_dpy  = 0;

    //dxR_dpy
    dxR_dpy  = 0;

    //dy_dpy
    dy_dpy  = fx/p_k_1[2];

    //Calculate related to J^T, py

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpy;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpy;

    dJ_dpy = dJ_dxL*dxL_dpy + dJ_dxR*dxR_dpy + dJ_dy*dy_dpy;
    dJT_dpy = dJT_dxL*dxL_dpy + dJT_dxR*dxR_dpy + dJT_dy*dy_dpy;

    Eigen::Matrix3d dUobs_dpy = Rwc*(dJ_dpy*Q*J.transpose() +J*Q*dJT_dpy)*Rwc.transpose();
    //cout<<"dUobs_dpy: "<<dUobs_dpy<<endl;

    //v = z
    //Calculate related to J
    //Calculate related to J, pz
    //equation (19) in paper
    double dxL_dpz, dxR_dpz, dy_dpz;

    double pz_2 = p_k_1[2] * p_k_1[2];
    //dxL_dpz
    dxL_dpz  = -fx*(p_k_1[0]+mb*0.5)/pz_2;

    //dxR_dpz
    dxR_dpz  = -fx*(p_k_1[0]-mb*0.5)/pz_2;

    //dy_dpz
    dy_dpz  = -fx*p_k_1[1]/pz_2;

    //Calculate related to J^T, pz

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpz;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpz;

    dJ_dpz = dJ_dxL*dxL_dpz + dJ_dxR*dxR_dpz + dJ_dy*dy_dpz;
    dJT_dpz = dJT_dxL*dxL_dpz + dJT_dxR*dxR_dpz + dJT_dy*dy_dpz;

    Eigen::Matrix3d dUobs_dpz = Rwc*(dJ_dpz*Q*J.transpose() +J*Q*dJT_dpz)*Rwc.transpose();
    //cout<<"dUobs_dpz: "<<dUobs_dpz<<endl;
    //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]
    dUpost_dpv << (front_Uobs*dUobs_dpx).trace(), (front_Uobs*dUobs_dpy).trace(), (front_Uobs*dUobs_dpz).trace();
    
    //cout<<"Gradient Test function Done"<<endl;
    return dUpost_dpv;
}

//Gradient of objective function respect to target position using cyclopean coordinate
Eigen::Vector3d SparsePipline::GradientForActiveKeypoint(const Eigen::Vector3d &input, const float &level, const Eigen::Matrix3d &U_prior, const Eigen::Matrix3d &Rwc)
{
    cout<<"Enter Gradient Test function"<<endl;
    cout<<"input: "<<input.transpose()<<endl;
    cout<<"level: "<<level<<endl;
    cout<<"U_prior: "<<U_prior<<endl;
    cout<<"Rwc: "<<Rwc<<endl;
    //transfer target's pixel coordinate to cyclopean coordinate
    float fx = mK.at<float>(0,0);
    float fy = mK.at<float>(1,1);
    float cx = mK.at<float>(0,2);
    float cy = mK.at<float>(1,2);

    Eigen::Vector3d p_k_1; //location at time k-1
    const float &xL = input[0] - cx;
    const float &xR = input[1] - cx;
    const float &y = input[2] - cy;  //pixel y
    
    double front_p = mb/(xL-xR);
    
    //active kepoint location in relative frame at time k-1 in cyclopean coordinate
    p_k_1 << front_p*0.5*(xL + xR), front_p*y, front_p*fx;

  
    Eigen::Matrix3d J = MakeJacobian(input);

    Eigen::Matrix3d Q = MakeQ(level);

    //partial differential of U_posterior at time k respect to [pk]v  dU_post/d[pk]v, v = x,y,z //target location
    Eigen::Vector3d dUpost_dpv;  //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]

    Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose();
 
    //cout<<"Uobs: "<<Uobs<<endl;
    //cout<<"U_prior: "<<U_prior<<endl;
    Eigen::Matrix3d U_post = (U_prior.inverse()+Uobs.inverse()).inverse();
   
    Eigen::Matrix3d U_post_2 = U_post*U_post;
    
    Eigen::Matrix3d front_Uobs = Uobs.inverse()*U_post_2*Uobs.inverse(); //front equation (16) and (10) in 2d paper

 
    //v = x
    //Calculate related to J, px
    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpx;
    //equation (19) in paper
    Eigen::Matrix3d dJ_dxL, dJ_dxR, dJ_dy;  //size: 3*3
    double dxL_dpx, dxR_dpx, dy_dpx;

    //common variable
    //(xL-xR)^3
    double xL_xR_3 = (xL-xR)*(xL-xR)*(xL-xR);
    //(xL-xR)^2
    double xL_xR_2 = (xL-xR)*(xL-xR);

    //dJ_dxL 
    dJ_dxL << 2*mb*xR/xL_xR_3, mb*(-xL-xR)/xL_xR_3, 0,
            2*mb*y/xL_xR_3, -2*mb*y/xL_xR_3, -mb/xL_xR_2,
            2*mb*fx/xL_xR_3, -2*mb*fx/xL_xR_3, 0;
    

    //dJ_dxR 
    dJ_dxR << -mb*(xR+xL)/xL_xR_3, 2*mb*xL/xL_xR_3, 0,
            -2*mb*y/xL_xR_3, 2*mb*y/xL_xR_3, mb/xL_xR_2,
            -2*mb*fx/xL_xR_3, 2*mb*fx/xL_xR_3, 0;
    

    //dJ_dy
    dJ_dy << 0, 0, 0,
            -mb/xL_xR_2, mb/xL_xR_2, 0,
            0, 0, 0;
    

    //Calculate related to J^T, px
    //dJT_px = dJT_dxL * dxL_dpx + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpx;
    Eigen::Matrix3d dJT_dxL, dJT_dxR, dJT_dy;  //size: 3*3

    //dJT_dxL
    dJT_dxL << 2*mb*xR/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*fx/xL_xR_3,
            mb*(-xL-xR)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*fx/xL_xR_3,
            0, -mb/xL_xR_2, 0;

    dJT_dxL = dJ_dxL.transpose();

    //dJT_dxR
    dJT_dxR << -mb*(xR+xL)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*fx/xL_xR_3,
            2*mb*xL/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*fx/xL_xR_3,
            0, mb/xL_xR_2, 0;
    dJT_dxR = dJ_dxR.transpose(); 
    //dJT_dy
    dJT_dy << 0, -mb/xL_xR_2, 0,
            0, mb/xL_xR_2, 0,
            0, 0, 0;
    dJT_dy = dJ_dy.transpose(); 
    
    //v = x
    //dxL_dpx
    dxL_dpx  = fx/p_k_1[2];

    //dxR_dpx
    dxR_dpx  = fx/p_k_1[2];

    //dy_dpx
    dy_dpx  = 0;

    dJ_dpx = dJ_dxL*dxL_dpx + dJ_dxR*dxR_dpx + dJ_dy*dy_dpx;
    dJT_dpx = dJT_dxL*dxL_dpx + dJT_dxR*dxR_dpx + dJT_dy*dy_dpx;

    Eigen::Matrix3d dUobs_dpx = Rwc*(dJ_dpx*Q*J.transpose() +J*Q*dJT_dpx)*Rwc.transpose();
    
    //cout<<"dUobs_dpx: "<<dUobs_dpx<<endl;
    //v = y
    //Calculate related to J, py
    //equation (19) in paper
    double dxL_dpy, dxR_dpy, dy_dpy;

    //dxL_dpy
    dxL_dpy  = 0;

    //dxR_dpy
    dxR_dpy  = 0;

    //dy_dpy
    dy_dpy  = fx/p_k_1[2];

    //Calculate related to J^T, py

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpy;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpy;

    dJ_dpy = dJ_dxL*dxL_dpy + dJ_dxR*dxR_dpy + dJ_dy*dy_dpy;
    dJT_dpy = dJT_dxL*dxL_dpy + dJT_dxR*dxR_dpy + dJT_dy*dy_dpy;

    Eigen::Matrix3d dUobs_dpy = Rwc*(dJ_dpy*Q*J.transpose() +J*Q*dJT_dpy)*Rwc.transpose();
    //cout<<"dUobs_dpy: "<<dUobs_dpy<<endl;

    //v = z
    //Calculate related to J
    //Calculate related to J, pz
    //equation (19) in paper
    double dxL_dpz, dxR_dpz, dy_dpz;

    double pz_2 = p_k_1[2] * p_k_1[2];
    //dxL_dpz
    dxL_dpz  = -fx*(p_k_1[0]+mb*0.5)/pz_2;

    //dxR_dpz
    dxR_dpz  = -fx*(p_k_1[0]-mb*0.5)/pz_2;

    //dy_dpz
    dy_dpz  = -fx*p_k_1[1]/pz_2;

    //Calculate related to J^T, pz

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpz;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpz;

    dJ_dpz = dJ_dxL*dxL_dpz + dJ_dxR*dxR_dpz + dJ_dy*dy_dpz;
    dJT_dpz = dJT_dxL*dxL_dpz + dJT_dxR*dxR_dpz + dJT_dy*dy_dpz;

    Eigen::Matrix3d dUobs_dpz = Rwc*(dJ_dpz*Q*J.transpose() +J*Q*dJT_dpz)*Rwc.transpose();
    //cout<<"dUobs_dpz: "<<dUobs_dpz<<endl;
    //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]
    dUpost_dpv << (front_Uobs*dUobs_dpx).trace(), (front_Uobs*dUobs_dpy).trace(), (front_Uobs*dUobs_dpz).trace();
    
    //cout<<"Gradient Test function Done"<<endl;
    return dUpost_dpv;
}


//test
Eigen::Vector3d SparsePipline::Gradient(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc)
{
    //cout<<"Enter Gradient Test function"<<endl;
    Eigen::Vector3d p_k_1; //location at time k-1
    float fx = mK.at<float>(0,0);
    float fy = mK.at<float>(1,1);
    float cx = mK.at<float>(0,2);
    float cy = mK.at<float>(1,2);
    
    const float &xL = input[0];
    const float &xR = input[1];
    const float &y = input[2];  //pixel y
    
    double front_p = mb/(xL-xR);
    
    //active kepoint location in relative frame at time k-1
    p_k_1 << front_p*(xL-cx), front_p*(y-cy), front_p*fx;

    //Jacobian matrix
    Eigen::Matrix3d J;
    J << cx-xR, xL-cx, 0,
            cy-y, y-cy, xL-xR,
            -fx, fx, 0;
    double front_J = mb/((xL - xR)*(xL - xR));
    
    J = front_J * J;

    int level = 8.0*depth/10.0;
    
    if(level >= 8)
        level = 7;

    Eigen::Matrix3d Q = MakeQ(level);


    //partial differential of U_posterior at time k respect to [pk]v  dU_post/d[pk]v, v = x,y,z //target location
    Eigen::Vector3d dUpost_dpv;  //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]

    Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose();
    //cout<<"Uobs: "<<Uobs<<endl;
    //cout<<"U_prior: "<<U_prior<<endl;
    Eigen::Matrix3d U_post = (U_prior.inverse()+Uobs.inverse()).inverse();
    Eigen::Matrix3d U_post_2 = U_post*U_post;
    Eigen::Matrix3d front_Uobs = Uobs.inverse()*U_post_2*Uobs.inverse(); //front equation (16) and (10) in 2d paper

    //cout<<"front_Uobs: "<<front_Uobs<<endl;

    //v = x
    //Calculate related to J, px
    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpx;
    //equation (19) in paper
    Eigen::Matrix3d dJ_dxL, dJ_dxR, dJ_dy;  //size: 3*3
    double dxL_dpx, dxR_dpx, dy_dpx;

    //common variable
    //(xL-xR)^3
    double xL_xR_3 = (xL-xR)*(xL-xR)*(xL-xR);
    //(xL-xR)^2
    double xL_xR_2 = (xL-xR)*(xL-xR);

    //dJ_dxL 
    dJ_dxL << -2*mb*(cx-xR)/xL_xR_3, mb*(-xL+2*cx-xR)/xL_xR_3, 0,
            2*mb*(y-cy)/xL_xR_3, -2*mb*(y-cy)/xL_xR_3, -mb/xL_xR_2,
            2*mb*fx/xL_xR_3, -2*mb*fx/xL_xR_3, 0;
    

    //dJ_dxR 
    dJ_dxR << mb*(-xR+2*cx-xL)/xL_xR_3, 2*mb*(xL-cx)/xL_xR_3, 0,
            -2*mb*(y-cy)/xL_xR_3, 2*mb*(y-cy)/xL_xR_3, mb/xL_xR_2,
            -2*mb*fx/xL_xR_3, 2*mb*fx/xL_xR_3, 0;
    

    //dJ_dy
    dJ_dy << 0, 0, 0,
            -mb/xL_xR_2, mb/xL_xR_2, 0,
            0, 0, 0;
    

    //Calculate related to J^T, px
    //dJT_px = dJT_dxL * dxL_dpx + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpx;
    Eigen::Matrix3d dJT_dxL, dJT_dxR, dJT_dy;  //size: 3*3

    //dJT_dxL
    dJT_dxL << -2*mb*(cx-xR)/xL_xR_3, 2*mb*(y-cy)/xL_xR_3, 2*mb*fx/xL_xR_3,
            mb*(-xL+2*cx-xR)/xL_xR_3, -2*mb*(y-cy)/xL_xR_3, -2*mb*fx/xL_xR_3,
            0, -mb/xL_xR_2, 0;

    //dJT_dxR
    dJT_dxR << mb*(-xR+2*cx-xL)/xL_xR_3, -2*mb*(y-cy)/xL_xR_3, -2*mb*fx/xL_xR_3,
            2*mb*(xL-cx)/xL_xR_3, 2*mb*(y-cy)/xL_xR_3, 2*mb*fx/xL_xR_3,
            0, mb/xL_xR_2, 0;

    //dJT_dy
    dJT_dy << 0, -mb/xL_xR_2, 0,
            0, mb/xL_xR_2, 0,
            0, 0, 0;

    
    //v = x
    //dxL_dpx
    dxL_dpx  = fx/p_k_1[2];

    //dxR_dpx
    dxR_dpx  = fx/p_k_1[2];

    //dy_dpx
    dy_dpx  = 0;

    dJ_dpx = dJ_dxL*dxL_dpx + dJ_dxR*dxR_dpx + dJ_dy*dy_dpx;
    dJT_dpx = dJT_dxL*dxL_dpx + dJT_dxR*dxR_dpx + dJT_dy*dy_dpx;

    Eigen::Matrix3d dUobs_dpx = Rwc*(dJ_dpx*Q*J.transpose() +J*Q*dJT_dpx)*Rwc.transpose();
    
    //cout<<"dUobs_dpx: "<<dUobs_dpx<<endl;
    //v = y
    //Calculate related to J, py
    //equation (19) in paper
    double dxL_dpy, dxR_dpy, dy_dpy;

    //dxL_dpy
    dxL_dpy  = 0;

    //dxR_dpy
    dxR_dpy  = 0;

    //dy_dpy
    dy_dpy  = fx/p_k_1[2];

    //Calculate related to J^T, py

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpy;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpy;

    dJ_dpy = dJ_dxL*dxL_dpy + dJ_dxR*dxR_dpy + dJ_dy*dy_dpy;
    dJT_dpy = dJT_dxL*dxL_dpy + dJT_dxR*dxR_dpy + dJT_dy*dy_dpy;

    Eigen::Matrix3d dUobs_dpy = Rwc*(dJ_dpy*Q*J.transpose() +J*Q*dJT_dpy)*Rwc.transpose();
    //cout<<"dUobs_dpy: "<<dUobs_dpy<<endl;

    //v = z
    //Calculate related to J
    //Calculate related to J, pz
    //equation (19) in paper
    double dxL_dpz, dxR_dpz, dy_dpz;

    double pz_2 = p_k_1[2] * p_k_1[2];
    //dxL_dpz
    dxL_dpz  =  -fx*p_k_1[0]/pz_2;

    //dxR_dpz
    dxR_dpz  = -fx*(p_k_1[0]-mb)/pz_2;

    //dy_dpz
    dy_dpz  = -fx*p_k_1[1]/pz_2;

    //Calculate related to J^T, pz

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpz;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpz;

    dJ_dpz = dJ_dxL*dxL_dpz + dJ_dxR*dxR_dpz + dJ_dy*dy_dpz;
    dJT_dpz = dJT_dxL*dxL_dpz + dJT_dxR*dxR_dpz + dJT_dy*dy_dpz;

    Eigen::Matrix3d dUobs_dpz = Rwc*(dJ_dpz*Q*J.transpose() +J*Q*dJT_dpz)*Rwc.transpose();
    //cout<<"dUobs_dpz: "<<dUobs_dpz<<endl;
    //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]
    dUpost_dpv << (front_Uobs*dUobs_dpx).trace(), (front_Uobs*dUobs_dpy).trace(), (front_Uobs*dUobs_dpz).trace();
    
    //cout<<"Gradient Test function Done"<<endl;
    return dUpost_dpv;
}


Eigen::Matrix3d SparsePipline::MakeJacobian(const Eigen::Vector3d &input)
{
    float fx = mK.at<float>(0,0);
    float fy = mK.at<float>(1,1);
    float cx = mK.at<float>(0,2);
    float cy = mK.at<float>(1,2);

    const float &xL = input[0];
    const float &xR = input[1];
    const float &y = input[2]; 

    

    Eigen::Matrix3d J;
    J << -xR, xL, 0,
            -y, y, xL-xR,
            -fx, fx, 0;
    double front_J = mb/((xL - xR)*(xL - xR));
    
    J = front_J * J;

    return J;
}

Eigen::Matrix3d SparsePipline::MakeQ(const int level)
{
    int nlevels = 8;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*1.2;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    const float &sigma2 = mvLevelSigma2[level];
    //cout<<"invSigma2 1: "<<invSigma2<<endl;
    
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity()*sigma2;

    return Q;
}


// vector<Eigen::Matrix3d> SparsePipline::ComputeObsCovarianceMatrix(const Eigen::Matrix3d &Rwc)
// {
//     int N = mCurrentFrame.N; //number of keypoints
    
//     float cx = mK.at<float>(0,2);
//     float cy = mK.at<float>(1,2);

//     for(int i = 0; i < N; ++i)
//     {
//         if(mCurrentFrame.mvuRight[i]<0) //this keypoint is occluded.
//             continue;
        
        
//         const cv::KeyPoint &kp = mCurrentFrame.mvKeys[i];

//         float uL = kp.pt.x;
//         float uR = mCurrentFrame.mvuRight[i];
//         float v = kp.pt.y;
//         int level = kp.octave;

//         //cyclopean coordinate
//         float xL = uL - cx;
//         float xR = uR - cx;
//         float y = v - cy;

//         Eigen::Vector3d input;
//         input << xL, xR, y;

//         //calculate covariance matrix of observation 

//         Eigen::Matrix3d J = MakeJacobian(input);
    
//         Eigen::Matrix3d Q = MakeQ(level);

//         Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose();

//         mCurrentFrame.mvKeyPointsObsCovariance[i] = Uobs;

//     }

//     return mCurrentFrame.mvKeyPointsObsCovariance;
// }

// void SparsePipline::UpdateCovarianceMatrix(const Frame &LastFrame, const Eigen::Matrix3d &Rwc)
// {
//     int N = mCurrentFrame.N; //number of keypoints

//     for(int i = 0; i < N; ++i)
//     {
//         if(mCurrentFrame.mvuRight[i]<0) //this keypoint is occluded.
//             continue;

//         //calculate covariance matrix of observation 
//         Eigen::Matrix3d Uobs  = mCurrentFrame.mvKeyPointsObsCovariance[i];

//         if(mCurrentFrame.mvmatchedNewKeypointsIndex[i]<0)  //this a new keypoint 
//         {
//             assert(Uobs != Eigen::Matrix3d::Zero());
//             Eigen::Matrix3d Upost = (Uobs.inverse()+Uobs.inverse()).inverse();
            
//             mCurrentFrame.mvKeyPointsPostCovariance[i] = Upost;

//         }
//         else  //this a not new keypoint
//         {
//             int matchedKpIndex = mCurrentFrame.mvmatchedNewKeypointsIndex[i];

//             Eigen::Matrix3d Uprior = LastFrame.mvKeyPointsPostCovariance[matchedKpIndex];
 
//             Eigen::Matrix3d Upost = (Uprior.inverse()+Uobs.inverse()).inverse();
            
//             mCurrentFrame.mvKeyPointsPostCovariance[i] = Upost;
//         }
//     }

// }

int SparsePipline::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void SparsePipline::WritePTCloud(const vector<Point<float>> &vPoints, const vector<float> &vDepth, const Delaunay<float> &delaunayTriangule)
{
    // ofstream fout("ptcloud.ply");
    // if(!fout.is_open())
    // {
    //     return ;
    // }
    // int num = vPoints.size();
    // int trian_num = delaunayTriangule.triangles.size();
    // fout << "ply" << endl;
    // fout << "format ascii 1.0" << endl;
    // fout <<"comment written by Weihan Wang" <<endl;
    // fout << "element vertex " << num << endl;
    // fout << "property float64 x" << endl;
    // fout << "property float64 y" << endl;
    // fout << "property float64 z" << endl;
    // //fout << "property uchar " << endl;
    // fout<<"element face "<< trian_num<<endl;
    // fout<<"property list uchar int vertex_indices"<<endl;
    // fout << "end_header" << endl;

    // //write 3D points in file
    // for(int row = 0; row < num; row++)
    // {
    //     const float zc = vDepth[row];

    //     const float xc = (vPoints[row].x - mCurrentFrame.cx)*zc*mCurrentFrame.invfx;
    //     const float yc = (vPoints[row].y - mCurrentFrame.cy)*zc*mCurrentFrame.invfy;

    //     cv::Mat x3Dc = (cv::Mat_<float>(3,1) << xc, yc, zc);

    //     cv::Mat x3Dw = mCurrentFrame.mRwc * x3Dc + mCurrentFrame.mtwc;

    //     const float x = x3Dw.at<float>(0,0);
    //     const float y =  x3Dw.at<float>(0,1);
    //     const float z = x3Dw.at<float>(0,2);

    //     uchar iC = mCurrentFrame.mpORBextractorLeft->mvImagePyramid[0].at<uchar>(vPoints[row].y, vPoints[row].x);
    //     fout << x << " " << y << " " << z <<endl;
    // }

    // //write triangule
    // for(Triangle<float> triangle: delaunayTriangule.triangles)
    // {
    //     Point<float> a;

    //     fout << 3 << " " << triangle.p0.nodeId << " " << triangle.p1.nodeId <<" " << triangle.p2.nodeId << endl;
    // }

    // fout.close();
}



}//end AVP