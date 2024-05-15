#pragma once
#include<vector>
#include<string>
#include<bits/stdc++.h>


std::string mapToId(std::string layerID, int channel, int row, int col){
	std::string res = layerID+"_"+
						std::to_string(channel)+"_"+
						std::to_string(row)+"_"+
						std::to_string(col)+"_";
	return res;
}

// std::vector< std::pair<std::string, std::vector<int>> >   mapToLayer(std::string layerID){
	
	

// }
template<typename T>
class ComputeNode
{
	public:
		std::string id;
		char operation;
		std::vector<ComputeNode *> fromNodes;
		ComputeNode *toNode;
		int indegree;

		T grad;
		bool grad_reached;

		ComputeNode(char operation_, std::string id_): operation(operation_), indegree(0), id(id_), grad(0.0), grad_reached(false){

		}
		void addFrom(ComputeNode *node){
			this -> fromNodes.push_back(node);
			this -> indegree++;
		}

		void assignTo(ComputeNode *node){
			this -> toNode = node;
		}
		
		void propagateGradients(){

		}
};