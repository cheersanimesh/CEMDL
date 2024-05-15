#pragma once
#include<vector>
#include<string>


std::string mapToId(std::string layerID, int channel, int row, int col){
	std::string res = layerID+"_"+
						std::to_string(channel)+"_"+
						std::to_string(row)+"_"+
						std::to_string(col)+"_";
	return res;
}

class GradientNode
{
	public:
		std::string id;
		char operation;
		std::vector<GradientNode *> fromNodes;
		std::vector<GradientNode *> toNodes;
		int indegree;

		GradientNode(char operation_, std::string id_): operation(operation_), indegree(0), id(id_)
		{

		}
		void addFrom(GradientNode *node){
			this -> fromNodes.push_back(node);
			this -> indegree++;
		}

		void assignTo(GradientNode *node){
			this -> toNode = node;
		}

};