#pragma once

#include "BoundingBox.hpp"
#include "Mask.hpp"

#include <vector>

class Metrics
{
public:
	Metrics(std::vector<BoundingBox> b, std::vector<Mask> m);
	~Metrics();

private:

};