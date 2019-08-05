#pragma once


class MedialAxis_C {

public : // Constructor/ Destructor 

	MedialAxis_C() = default;


public : // Public Methods

	void GetForegroundImage();

	void ApplyMorphologicalOperator();

	void ApplyGradient();

	void ApplyHoughTransform();

	void GetHoughLine();

	void GetEdge();
	
	void GetMedialAxis();


private: // internal methods


private : // class attributes


};