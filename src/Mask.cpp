#include "Mask.hpp"

Mask::Mask(const cv::Mat i, BoundingBox b)
	: source_image(i), bb(b)
{
}

Mask::~Mask()
{
}