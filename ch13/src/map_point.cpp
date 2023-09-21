// map_point.cpp
// 2023 JUL 03
// Tershire reformatted the code...

// created by gaoxiang.

#include "my_VO/map_point.h"
#include "my_VO/feature.h"

namespace my_VO
{

MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

MapPoint::Ptr MapPoint::CreateNewMappoint()
{
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat)
{
    // std::cout << "remove_observation ==========================" << std::endl;
    // std::cout << "observed_times_: " << observed_times_ << std::endl;
    // std::cout << "is feature null: " << (feat == nullptr) << std::endl;
    // std::cout << "observations_.size(): " << observations_.size() << std::endl;
    
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end();
         iter++)
    {
        if (iter->lock() == feat)
        {
            observations_.erase(iter);
            feat->map_point_.reset();
            observed_times_--;
            break;
        }
    }
}

}  // namespace my_VO
