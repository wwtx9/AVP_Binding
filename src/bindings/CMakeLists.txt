pybind11_add_module(AVP_Binding SHARED
        bindings.cpp
        ../app/Frame.cpp
        ../app/SparsePipeline.cpp
        ../app/System.cpp
        ../app/ORBextractor.cpp
        ../app/Converter.cpp
        ../app/Delaunay.hpp
        ../app/Visualization.cpp
        ../app/Mat_Wraper.cpp
        )
target_link_libraries(AVP_Binding PRIVATE
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )
