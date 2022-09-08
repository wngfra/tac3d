// Copyright (C) 2022 wngfra/captjulian
// 
// tac3d is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// tac3d is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with tac3d. If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "control_interfaces/srv/control2d.hpp"

namespace tac3d
{
    class MotionController : public rclcpp::Node
    {
    public:
        MotionController(const std::string service_name);
        //~MotionController();
        void sendControlRequest(const float dx, const float dy);

        rclcpp::Node::SharedPtr m_node;
    private:
        void tactilePublisherCallback(const sensor_msgs::msg::Image::SharedPtr msg);

        rclcpp::Client<control_interfaces::srv::Control2d>::SharedPtr m_client;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr tactile_sub;
    };
}  // namespace tac3d