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
#include <sensor_msgs/msg/Image.hpp>

#include "control_interfaces/srv/control2_d.hpp"

namespace tac3d
{
    class MotionControlClient
    {
    public:
        MotionControlClient(std::string node_name, std::string service_name);
        ~MotionControlClient();

        void setVelocity(double x, double y, double theta);
        void setVelocity(double x, double y);
        void setVelocity(double theta);

    private:
        rclcpp::Node::SharedPtr m_node;
        rclcpp::Client<control_interfaces::srv::Control2D>::SharedPtr m_client;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr tactile_sub;
    };
}  // namespace tac3d