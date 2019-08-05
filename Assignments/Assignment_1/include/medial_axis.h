#pragma once


class MedialAxis_C {

public : // Constructor/ Destructor 

	MedialAxis_C() = default;


public : // Public Methods


	void ApplyHoughTransform();

	void GetHoughLine();

	void GetEdge();

	void GetForegroundImage();

	void GetMedialAxis();

};