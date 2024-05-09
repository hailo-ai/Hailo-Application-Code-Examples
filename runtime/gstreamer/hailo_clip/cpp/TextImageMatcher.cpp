#include "TextImageMatcher.hpp"
// Define static members
TextImageMatcher* TextImageMatcher::instance = nullptr;
std::mutex TextImageMatcher::mutex;