// config.cpp
// 2023 JUL 03
// Tershire reformatted the code...

// created by gaoxiang.

#include "my_VO/config.h"

namespace my_VO
{

bool Config::SetParameterFile(const std::string &filename)
{
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->file_.isOpened() == false)
    {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        config_->file_.release();
        return false;
    }
    return true;
}

Config::~Config()
{
    if (file_.isOpened())
        file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

} // namespace my_VO
