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
    mb = mbf/fx;

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

    vector<Point<float>> vPoints;
    vector<float> vDepth;

    //SLAM method
    Eigen::Matrix4d currentPose = Track();
    mRwc = currentPose.block<3,3>(0,0);
    mtwc = currentPose.block<3,1>(0,3);

    //Step 2: Apply Delaunay triangulation on the keypoints
    DelaunayTriangulation(vPoints, vDepth);

    // Step 3: Select as the active keypoint the one with smallest expected reconstruction error
    int activekpIndex = SelectActiveKeyPoint();
    cout<<activekpIndex<<endl;
    //Step 4: Compute the NBV for the active keypoint

    //Step 4.1 Determine the next best target location in the relative frame
    //Eigen::Vector3d NextBestTargetLocation(Eigen::Vector3d &p_k, Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior);


    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for(int iL = 0; iL < mCurrentFrame.N; iL++)
    {
        if(mCurrentFrame.mvDepth[iL]>0)
            vDepthIdx.push_back({mCurrentFrame.mvDepth[iL], iL});
    }
    sort(vDepthIdx.begin(), vDepthIdx.end(),[](const pair<float, int>  &a, const pair<float, int>  &b){return a.first< b.first;});

    cv::Mat im, im2;
    imRectLeft.copyTo(im);
    imRectRight.copyTo(im2);
    cvtColor(im, im, CV_GRAY2RGB);
    cvtColor(im2, im2, CV_GRAY2RGB);

    const float r = 5;


    //end
    for(int i = 0; i < vDepthIdx.size(); i++)
    {
        if(vDepthIdx[i].second == activekpIndex)
        {
            cv::Point2f pt1,pt2; //float
            pt1.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-r;
            pt1.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
            pt2.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x+r;
            pt2.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;

            cout<<"active keypoint depth (other color): "<< vDepthIdx[i].first<<endl;
            cv::rectangle(im,pt1,pt2,cv::Scalar(120,120,120)); // green close
            cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(120,120,120),-1);

        }

        if (i==0 || i == vDepthIdx.size()/2 || i == vDepthIdx.size()-1){
            cv::Point2f pt1,pt2;
            pt1.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-r;
            pt1.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
            pt2.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x+r;
            pt2.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;

            //cv::Point2f pt1R,pt2R;
            double dispar = mCurrentFrame.mbf/vDepthIdx[i].first;
            //pt1R.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar-r;
            //pt1R.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y-r;
            //pt2R.x=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar+r;
            //pt2R.y=mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y+r;
            cv::Point2f cur(cv::Point2f(mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.x-dispar, mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt.y));

            if (i == 0){
                //cout<<"close dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
                cout<<"close depth (green): "<< vDepthIdx[i].first<<endl;
                cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0)); // green close
                cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(0,255,0),-1);

                //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(0,255,0)); // green close
                //cv::circle(im2,cur,2,cv::Scalar(0,255,0),-1);

            } else if (i == vDepthIdx.size()/2){
                //cout<<"between dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
                cout<<"between depth (blue): "<< vDepthIdx[i].first<<endl;
                cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));   // blue
                cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(255,0,0),-1);

                //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(255,0,0));   // blue
                //cv::circle(im2,cur,2,cv::Scalar(255,0,0),-1);
            } else {
                //cout<<"far dispar: "<< mCurrentFrame.mbf/vDepthIdx[i].first<<endl;
                cout<<"far depth (red): "<< vDepthIdx[i].first<<endl;
                cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,255));   // red far
                cv::circle(im,mCurrentFrame.mvKeysUn[vDepthIdx[i].second].pt,2,cv::Scalar(0,0,255),-1);

                //cv::rectangle(im2,pt1R,pt2R,cv::Scalar(0,0,255));   // red far
                //cv::circle(im2,cur,2,cv::Scalar(0,0,255),-1);

            }


        }
    }

//    cv::Point2f pt1,pt2;
//
//    pt1.x=mCurrentFrame.mvKeysUn[activekpIndex].pt.x-r;
//    pt1.y=mCurrentFrame.mvKeysUn[activekpIndex].pt.y-r;
//    pt2.x=mCurrentFrame.mvKeysUn[activekpIndex].pt.x+r;
//    pt2.y=mCurrentFrame.mvKeysUn[activekpIndex].pt.y+r;
//    cout<<"activekpIndex: "<< activekpIndex<<endl;
//    cout<<"Depth of this kp: "<<mCurrentFrame.mvDepth[activekpIndex]<<endl;
//    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
//    cv::circle(im,mCurrentFrame.mvKeysUn[activekpIndex].pt,2,cv::Scalar(0,255,0),-1);
    cout<<"activekp index: "<<activekpIndex<<", close kp index: "<<vDepthIdx[0].second<<endl;
    //cv::imshow("AVP: Current Frame ", im);
    //cv::imshow("AVP: Current Right Frame", im2);
    //cv::waitKey(0);

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

int SparsePipline::SelectActiveKeyPoint()
{
    int activeKeyPointIndex = -1;


    double min_trace = 1.0*INT_MAX;

    for(size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
    {


        if(mCurrentFrame.mvDepth[i]<0)
            continue;
        const cv::KeyPoint &kpUn = mCurrentFrame.mvKeysUn[i];
        float xL = kpUn.pt.x, xR = mCurrentFrame.mvuRight[i], y = kpUn.pt.y;

        const float &invSigma2 = mCurrentFrame.mvInvLevelSigma2[kpUn.octave]; //mCurrentFrame.mvDepth[i]*mCurrentFrame.mvDepth[i];
        Eigen::Matrix3d Q = Eigen::Matrix3d::Identity()*invSigma2;

        Eigen::Matrix3d J;
        J << -xR, xL, 0,
                -y, y, xL-xR,
                -mCurrentFrame.fx, mCurrentFrame.fx, 0;
        double front = mb/((xL - xR)*(xL - xR));
        J = front * J;

        Eigen::Matrix3d Cov = J*Q *J.transpose();

        cv::Mat R = cv::Mat::eye(4,4,CV_32F);
        //mCurrentFrame.GetPoseInverse().rowRange(0,3).colRange(0,3); // Rwc
        Eigen::Matrix3d R_eigen = Converter::toMatrix3d(R);

        Cov = R_eigen * Cov * R_eigen.transpose();

        double currentTrace = Cov(0,0) + Cov(1,1) + Cov(2,2);


        if(currentTrace < min_trace)
        {
            min_trace = currentTrace;
            activeKeyPointIndex = i;
        }
    }

    return activeKeyPointIndex;
}

Eigen::Vector3d SparsePipline::GradientBasedOnNextBestTargetLocation(const int &kpIndex, Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc)
{
    Eigen::Vector3d p_k_1; //location at time k-1
    double deltaT;
    const cv::KeyPoint &activekpUn = mCurrentFrame.mvKeysUn[kpIndex];

    const float &xL = activekpUn.pt.x;
    const float &xR = mCurrentFrame.mvuRight[kpIndex];
    const float &y = activekpUn.pt.y;  //pixel y

    double front_p = mb/(xL-xR);
    //active kepoint location in relative frame at time k-1
    p_k_1 << front_p*(xL+xR)*0.5, front_p*y, front_p*mCurrentFrame.fx;

    //Jacobian matrix
    Eigen::Matrix3d J;
    J << -xR, xL, 0,
            -y, y, xL-xR,
            -mCurrentFrame.fx, mCurrentFrame.fx, 0;
    double front_J = mb/((xL - xR)*(xL - xR));
    J = front_J * J;

    //Q covariance matrix for pixel
    const float &invSigma2 = mCurrentFrame.mvInvLevelSigma2[activekpUn.octave];
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity()*invSigma2;

    //partial differential of U_posterior at time k respect to [pk]v  dU_post/d[pk]v, v = x,y,z //target location
    Eigen::Vector3d dUpost_dpv;  //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]

    Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose(); 
 
    Eigen::Matrix3d front_Uobs = Uobs.inverse()*(U_prior.inverse()+Uobs.inverse())*(U_prior.inverse()+Uobs.inverse())*Uobs.inverse(); //front equation (16) and (10) in 2d paper

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

    //dJ_dxL and dxL_dpx
    dJ_dxL << 2*mb*xR/xL_xR_3, mb*(-xR-xL)/xL_xR_3, 0,
            2*mb*y/xL_xR_3, -2*mb*y/xL_xR_3, -mb/xL_xR_2,
            2*mb*mCurrentFrame.fx/xL_xR_3, -2*mb*mCurrentFrame.fx/xL_xR_3, 0;
    dxL_dpx  = mCurrentFrame.fx/p_k_1[2];

    //dJ_dxR and dxR_dpx
    dJ_dxR << -mb*(xR+xL)/xL_xR_3, 2*mb*xL/xL_xR_3, 0,
            -2*mb*y/xL_xR_3, 2*mb*y/xL_xR_3, mb/xL_xR_2,
            -2*mb*mCurrentFrame.fx/xL_xR_3, 2*mb*mCurrentFrame.fx/xL_xR_3, 0;
    dxR_dpx  = mCurrentFrame.fx/p_k_1[2];

    //dJ_dy and dy_dpx
    dJ_dy << 0, 0, 0,
            -mb/xL_xR_2, mb/xL_xR_3, 0,
            0, 0, 0;
    dy_dpx  = 0;

    //Calculate related to J^T, px
    //dJT_px = dJT_dxL * dxL_dpx + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpx;
    Eigen::Matrix3d dJT_dxL, dJT_dxR, dJT_dy;  //size: 3*3

    //dJT_dxL
    dJT_dxL << 2*mb*xR/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*mCurrentFrame.fx/xL_xR_3,
            -mb*(xR-xL)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*mCurrentFrame.fx/xL_xR_3,
            0, -mb/xL_xR_2, 0;

    //dJT_dxR
    dJT_dxR << -mb*(xR+xL)/xL_xR_3, -2*mb*y/xL_xR_3, -2*mb*mCurrentFrame.fx/xL_xR_3,
            2*mb*xL/xL_xR_3, 2*mb*y/xL_xR_3, 2*mb*mCurrentFrame.fx/xL_xR_3,
            0, mb/xL_xR_2, 0;

    //dJT_dy
    dJT_dy << 0, -mb/xL_xR_2, 0,
            0, mb/xL_xR_2, 0,
            0, 0, 0;

    dJ_dpx = dJ_dxL*dxL_dpx + dJ_dxR*dxR_dpx + dJ_dy*dy_dpx;
    dJT_dpx = dJT_dxL*dxL_dpx + dJT_dxR*dxR_dpx + dJT_dy*dy_dpx;

    Eigen::Matrix3d dUobs_dpx = Rwc*(dJ_dpx*Q*J.transpose() +J*Q*dJT_dpx)*Rwc.transpose();

    //v = y
    //Calculate related to J, py
    //equation (19) in paper
    double dxL_dpy, dxR_dpy, dy_dpy;

    //dxL_dpy
    dxL_dpy  = 0;

    //dxR_dpy
    dxR_dpy  = 0;

    //dy_dpy
    dy_dpy  = mCurrentFrame.fx/p_k_1[2];

    //Calculate related to J^T, py

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpy;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpy;

    dJ_dpy = dJ_dxL*dxL_dpy + dJ_dxR*dxR_dpy + dJ_dy*dy_dpy;
    dJT_dpy = dJT_dxL*dxL_dpy + dJT_dxR*dxR_dpy + dJT_dy*dy_dpy;

    Eigen::Matrix3d dUobs_dpy = Rwc*(dJ_dpy*Q*J.transpose() +J*Q*dJT_dpy)*Rwc.transpose();

    //v = z
    //Calculate related to J
    //Calculate related to J, pz
    //equation (19) in paper
    double dxL_dpz, dxR_dpz, dy_dpz;

    double pz_2 = p_k_1[2] * p_k_1[2];
    //dxL_dpz
    dxL_dpy  = -mCurrentFrame.fx*(p_k_1[0]+0.5*mb)/pz_2;

    //dxR_dpz
    dxR_dpz  = -mCurrentFrame.fx*(p_k_1[0]-mb*0.5)/pz_2;

    //dy_dpz
    dy_dpz  = -mCurrentFrame.fx*p_k_1[1]/pz_2;

    //Calculate related to J^T, pz

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dpz;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dpz;

    dJ_dpz = dJ_dxL*dxL_dpz + dJ_dxR*dxR_dpz + dJ_dy*dy_dpz;
    dJT_dpz = dJT_dxL*dxL_dpz + dJT_dxR*dxR_dpz + dJT_dy*dy_dpz;

    Eigen::Matrix3d dUobs_dpz = Rwc*(dJ_dpz*Q*J.transpose() +J*Q*dJT_dpz)*Rwc.transpose();

    //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]
    dUpost_dpv << (front_Uobs*dUobs_dpx).trace(), (front_Uobs*dUobs_dpy).trace(), (front_Uobs*dUobs_dpz).trace();

    // Eigen::Vector3d pk;

    // pk -= pk-dUpost_dpv*deltaT;
    return dUpost_dpv;
}

Eigen::Vector3d SparsePipline::GradientofCameraTranslation(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc)
{
        //cout<<"Enter Gradient Test function"<<endl;
    Eigen::Vector3d p_k_1; //location at time k-1
    float cx = 960/2.0;
    float cy = 640/2.0;
    const float &xL = input[0];
    const float &xR = input[1];
    const float &y = input[2];  //pixel y
    //cout<<xL<<" ,"<<xR<<" ,"<<y<<endl;
    double fx = 450.0;
    mb = 0.2;
    double front_p = mb/(xL-xR);
    
    //active kepoint location in relative frame at time k-1
    p_k_1 << front_p*(xL+xR)*0.5, front_p*y, front_p*fx;
    
    //Jacobian matrix
    Eigen::Matrix3d J;
    J << cx-xR, xL-cx, 0,
            cy-y, y-cy, xL-xR,
            -fx, fx, 0;
    double front_J = mb/((xL - xR)*(xL - xR));
    
    J = front_J * J;
    
    //Q covariance matrix for pixel
    int level = 8.0*depth/10.0;
   // cout<<"level: "<<level<<endl;
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

    const float &invSigma2 = mvInvLevelSigma2[level];
    //cout<<"invSigma2 1: "<<invSigma2<<endl;
    
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity()*invSigma2;
   // cout<<"invSigma2: "<<invSigma2<<endl;

    //partial differential of U_posterior at time k respect to [pk]v  dU_post/d[pk]v, v = x,y,z //target location
    Eigen::Vector3d dUpost_dtcw;  //dUpost_dtcw = [tr(front*dUobs_dtcw[0]), tr(front*dUobs_dtcw[1]),tr(front*dUobs_dtcw[2])]

    Eigen::Matrix3d Uobs  = Rwc * J * Q * J.transpose() * Rwc.transpose();
    //cout<<"Uobs: "<<Uobs<<endl;
    //cout<<"U_prior: "<<U_prior<<endl;
    Eigen::Matrix3d U_post = (U_prior.inverse()+Uobs.inverse()).inverse();
    Eigen::Matrix3d U_post_2 = U_post*U_post;
    Eigen::Matrix3d front_Uobs = Uobs.inverse()*U_post_2*Uobs.inverse(); //front equation (16) and (10) in 2d paper

    //cout<<"front_Uobs: "<<front_Uobs<<endl;

   
    //Calculate related to J, px
    //dJ_px = dJ_dxL * dxL_dpx * dpx_dtcw0 + dJ_dxR * dxR_dpx * dpx_dtcw0 + dJ_dy * dy_dpx * dpx_dtcw0
    Eigen::Matrix3d dJ_dtcw0;
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
    Eigen::Matrix3d dJT_dtcw0;
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

    
    //u = 0
    //dxL_dpx
    dxL_dpx  = fx/p_k_1[2];

    //dxR_dpx
    dxR_dpx  = fx/p_k_1[2];

    //dy_dpx
    dy_dpx  = 0;

    dJ_dtcw0 = dJ_dxL*dxL_dpx + dJ_dxR*dxR_dpx+ dJ_dy*dy_dpx;
    dJT_dtcw0 = dJT_dxL*dxL_dpx + dJT_dxR*dxR_dpx + dJT_dy*dy_dpx;

    Eigen::Matrix3d dUobs_dtcw0 = Rwc*(dJ_dtcw0*Q*J.transpose() +J*Q*dJT_dtcw0)*Rwc.transpose();
    
    //u =1
    double dxL_dpy, dxR_dpy, dy_dpy;

    //dxL_dpy
    dxL_dpy  = 0;

    //dxR_dpy
    dxR_dpy  = 0;

    //dy_dpy
    dy_dpy  = fx/p_k_1[2];

    //Calculate related to J^T, py

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dtcw1;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dtcw1;

    dJ_dtcw1 = dJ_dxL*dxL_dpy + dJ_dxR*dxR_dpy + dJ_dy*dy_dpy;
    dJT_dtcw1 = dJT_dxL*dxL_dpy + dJT_dxR*dxR_dpy + dJT_dy*dy_dpy;

    Eigen::Matrix3d dUobs_dtcw1 = Rwc*(dJ_dtcw1*Q*J.transpose() +J*Q*dJT_dtcw1)*Rwc.transpose();
    //cout<<"dUobs_dpy: "<<dUobs_dpy<<endl;

    //u = 2
    double dxL_dpz, dxR_dpz, dy_dpz;

    double pz_2 = p_k_1[2] * p_k_1[2];
    //dxL_dpz
    dxL_dpz  = -fx*p_k_1[0]/pz_2;

    //dxR_dpz
    dxR_dpz  = -fx*(p_k_1[0]-mb)/pz_2;

    //dy_dpz
    dy_dpz  = -fx*p_k_1[1]/pz_2;

    //Calculate related to J^T, pz

    //dJ_px = dJ_dxL * dxL_dpx + dJ_dxR * dxR_dpx + dJ_dy * dy_dpx equation (18) in paper
    Eigen::Matrix3d dJ_dtcw2;
    //dJT_py = dJT_dxL * dxL_dpy + dJT_dxR * dxR_dpx + dJT_dy * dy_dpx;
    Eigen::Matrix3d dJT_dtcw2;

    dJ_dtcw2 = dJ_dxL*dxL_dpz + dJ_dxR*dxR_dpz + dJ_dy*dy_dpz;
    dJT_dtcw2 = dJT_dxL*dxL_dpz + dJT_dxR*dxR_dpz + dJT_dy*dy_dpz;

    Eigen::Matrix3d dUobs_dtcw2 = Rwc*(dJ_dtcw2*Q*J.transpose() +J*Q*dJT_dtcw2)*Rwc.transpose();
    //cout<<"dUobs_dpz: "<<dUobs_dpz<<endl;
    //dUpost_dpv = [tr(front*dUobs_dpx), tr(front*dUobs_dpy),tr(front*dUobs_dpz)]
    dUpost_dtcw << (front_Uobs*dUobs_dtcw0).trace(), (front_Uobs*dUobs_dtcw1).trace(), (front_Uobs*dUobs_dtcw2).trace();
    
    //cout<<"Gradient Test function Done"<<endl;
    return dUpost_dtcw;
}

//test
Eigen::Vector3d SparsePipline::Gradient(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc)
{
    //cout<<"Enter Gradient Test function"<<endl;
    Eigen::Vector3d p_k_1; //location at time k-1

    float cx = 960/2.0;
    float cy = 640/2.0;
    const float &xL = input[0];
    const float &xR = input[1];
    const float &y = input[2];  //pixel y
    //cout<<xL<<" ,"<<xR<<" ,"<<y<<endl;
    double fx = 450.0;
    mb = 0.2;
    double front_p = mb/(xL-xR);
    
    //active kepoint location in relative frame at time k-1
    p_k_1 << front_p*(xL+xR)*0.5, front_p*y, front_p*fx;
    
    //Jacobian matrix
    Eigen::Matrix3d J;
    J << cx-xR, xL-cx, 0,
            cy-y, y-cy, xL-xR,
            -fx, fx, 0;
    double front_J = mb/((xL - xR)*(xL - xR));
    
    J = front_J * J;
    
    //Q covariance matrix for pixel
    int level = 8.0*depth/10.0;
   // cout<<"level: "<<level<<endl;
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

    const float &invSigma2 = mvInvLevelSigma2[level];
    //cout<<"invSigma2 1: "<<invSigma2<<endl;
    
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity()*invSigma2;
   // cout<<"invSigma2: "<<invSigma2<<endl;

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
    dxL_dpz  = -fx*p_k_1[0]/pz_2;

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


void SparsePipline::WritePTCloud(const vector<Point<float>> &vPoints, const vector<float> &vDepth, const Delaunay<float> &delaunayTriangule)
{
    ofstream fout("ptcloud.ply");
    if(!fout.is_open())
    {
        return ;
    }
    int num = vPoints.size();
    int trian_num = delaunayTriangule.triangles.size();
    fout << "ply" << endl;
    fout << "format ascii 1.0" << endl;
    fout <<"comment written by Weihan Wang" <<endl;
    fout << "element vertex " << num << endl;
    fout << "property float64 x" << endl;
    fout << "property float64 y" << endl;
    fout << "property float64 z" << endl;
    //fout << "property uchar " << endl;
    fout<<"element face "<< trian_num<<endl;
    fout<<"property list uchar int vertex_indices"<<endl;
    fout << "end_header" << endl;

    //write 3D points in file
    for(int row = 0; row < num; row++)
    {
        const float zc = vDepth[row];

        const float xc = (vPoints[row].x - mCurrentFrame.cx)*zc*mCurrentFrame.invfx;
        const float yc = (vPoints[row].y - mCurrentFrame.cy)*zc*mCurrentFrame.invfy;

        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << xc, yc, zc);

        cv::Mat x3Dw = mCurrentFrame.mRwc * x3Dc + mCurrentFrame.mtwc;

        const float x = x3Dw.at<float>(0,0);
        const float y =  x3Dw.at<float>(0,1);
        const float z = x3Dw.at<float>(0,2);

        uchar iC = mCurrentFrame.mpORBextractorLeft->mvImagePyramid[0].at<uchar>(vPoints[row].y, vPoints[row].x);
        fout << x << " " << y << " " << z <<endl;
    }

    //write triangule
    for(Triangle<float> triangle: delaunayTriangule.triangles)
    {
        Point<float> a;

        fout << 3 << " " << triangle.p0.nodeId << " " << triangle.p1.nodeId <<" " << triangle.p2.nodeId << endl;
    }

    fout.close();
}



}//end AVP