# NEXUS: A Community Detection Platform

## Introduction

Social media platforms have become integral to our daily lives, generating vast datasets of user interactions. Analyzing these datasets is crucial for various purposes, including marketing strategies, user engagement, and understanding information dissemination dynamics. This project focuses on developing a cutting-edge web-based platform tailored for big companies seeking to identify, engage, and collaborate with influencers from their targeted communities.

## Community Detection

Community detection in social network analysis plays a pivotal role in identifying clusters and communities within these networks, helping to unveil hidden patterns and influential nodes.

### Definition

A community, with respect to graphs, can be defined as a subset of nodes that are densely connected to each other and loosely connected to the nodes in other communities in the same graph.

### Graph Theory

Graph theory is used to represent the social network as a mathematical graph, where users or entities are represented as nodes, and their connections or interactions are represented as edges.

## Influence Maximization

Influence maximization aims to identify a subset of individuals (nodes) in a social network that can maximize the spread of influence, such as information, ideas, or trends. This concept has applications in viral marketing, public health campaigns, and social network analysis.

## Algorithms Used

- **Artificial Bee Colony**: A nature-based optimization algorithm.
- **Louvain Algorithm**: Maximizes modularity in network structures.
- **Maxmin Algorithm**: Maximizes the minimum intra-community similarity.
- **Label Propagation Algorithm**: Assigns labels to nodes based on their neighbors.
- **Degree Centrality**: Measures influence based on the number of connections.

## Problem Statement

In today's digital landscape, understanding and leveraging the dynamics of online communities is paramount for big companies aiming to establish meaningful connections with their target audience and influencers. The problem addressed by this project is to develop a web-based platform tailored for big companies seeking to identify, engage, and collaborate with influencers from their targeted communities.

## Significance & Novelty

### Significance

- **Influencer Identification**: Implement advanced community detection algorithms to accurately identify influential nodes.
- **Community Insights**: Provide insights into the structure, behavior, and sentiment of targeted communities.
- **Engagement Strategy**: Develop tools and strategies for meaningful engagement with influencers.
- **Performance Analytics**: Incorporate performance metrics and analytics dashboards to track the effectiveness of influencer engagement campaigns.
- **Scalability and Integration**: Ensure the platform is scalable and interoperable with existing marketing systems.

### Novelty

- **Growing Importance of Online Communities**: Addressing the complexity of navigating and leveraging large-scale online communities.
- **Complexity of Influencer Identification**: Moving beyond simple popularity metrics to understand network structures and influence dynamics.
- **Integration of Advanced Technologies**: Utilizing sophisticated tools that integrate advanced network analysis and community detection techniques.
- **Data Volume and Variety**: Handling vast amounts of network data from various social media platforms and digital channels.
- **Uncovering Hidden Patterns**: Identifying influential nodes and cohesive communities within network data.
- **Tailored Solutions for Big Companies**: Developing a platform specifically for big companies to enhance their influencer marketing strategies.
- **Empowerment through Insights**: Providing actionable insights for informed decision-making and strategic collaboration.

## Solution Approach

This application is a web-based platform where users input desired influencers, select community detection algorithms, sorting criteria, and a CSV dataset. The platform visualizes the dataset as a graph and implements community detection algorithms to identify cohesive communities. Users receive tables containing community nodes, seed nodes, and influenced nodes, alongside visualizations where seed nodes are highlighted.

### Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Python
- **Libraries**: NetworkX, Matplotlib, Community

## Requirement Analysis

### Functional Requirements

- **Community Detection Framework**: Support community detection and influence maximization strategies.
- **Community Detection Algorithms**: Implement Artificial Bee Colony, Louvain, Maxmin, Label Propagation, and Degree Centrality algorithms.
- **Visualization**: Provide graphical representations of the social network and algorithm iterations.
- **User Interaction**: Allow users to input parameters and explore results.
- **Data Handling**: Efficiently read and process social network data from CSV files.

### Non-Functional Requirements

- **Performance**: Ensure efficient algorithm execution and minimize response times.
- **Scalability**: Design the application to handle growing datasets and user loads.
- **Reliability**: Implement error handling and regular testing.
- **Security**: Secure user inputs and protect the application.
- **Usability**: Design an intuitive user interface.
- **Compatibility**: Ensure compatibility with various browsers and devices.

### Logical Database Requirements

- **Data Storage**: Store social network data for community detection.
- **Querying**: Enable efficient querying for algorithm execution.
- **Data Integrity**: Ensure data integrity and regular validation.
- **Backup and Recovery**: Implement regular backups and recovery procedures.

## Implementation Details

The program is broken down into major components: input handling, algorithm implementation, visualization, and user interaction.

### Input

- **CSV File**: Contains information about connections between nodes.
- **Budget**: Number of seed nodes to be selected.
- **Algorithm Selection**: Choose from Artificial Bee Colony, Louvain, Maxmin, Label Propagation.

### Algorithms Used

- **Artificial Bee Colony**: Optimizes community detection by simulating bee foraging behavior.
- **Louvain Algorithm**: Identifies communities by optimizing modularity.
- **Maxmin Algorithm**: Maximizes the minimum similarity between nodes within communities.
- **Label Propagation Algorithm**: Assigns labels to nodes based on neighbors' labels.
- **Degree Centrality**: Identifies influential nodes based on the number of connections.

### Key Actors and Use Cases

- **User**: Interacts with the web application.
- **Use Cases**: Enter parameters, upload CSV file, select sorting method, submit form, display visualizations and results.

### Implementation Challenges

- **Algorithm Efficiency**: Ensuring efficient execution with large datasets.
- **Scalability**: Handling increasing data volumes and user loads.
- **Usability**: Designing an intuitive and user-friendly interface.
- **Data Handling**: Efficiently processing and analyzing large network datasets.

## Running the Project

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/social-media-community-detection.git
   cd social-media-community-detection

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt

3. Run the application:
   ```sh
   flask run

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contributing

- Fork the repository:

1. Click the "Fork" button at the top right of this page.

2. Create a branch:

```sh
git checkout -b feature/shubhaygautam
```
3. Commit your changes:
   ```sh
   git commit -m 'Add some feature'

4. Push to the branch:
   ```sh
   git push origin feature/shubhaygautam

5. Create a Pull Request:
- Open a pull request to merge your changes into the main branch.

## Output Screenshots
- Main Page
 ![image](https://github.com/shubhaygautam/Nexus---A-Community-Detection-Platform/assets/111029251/db7584eb-ce55-4de2-ba37-20f6ed79880e)



 
- Selecting an algorithm:
  ![image](https://github.com/shubhaygautam/NEXUS-A-Community-Detection-Platform/assets/111029251/01a15de1-853a-4893-8dc0-da9f20b9ac30)




- Time taken by the algorithm:

  ![image](https://github.com/shubhaygautam/NEXUS-A-Community-Detection-Platform/assets/111029251/00f1b504-4e9e-4d36-9b06-b8a22c601471)



- Visualized Graph with communities:
  
  ![image](https://github.com/shubhaygautam/NEXUS-A-Community-Detection-Platform/assets/111029251/2cb21cc6-2430-4a0a-bfaf-4882829f1ddb)



- Connection Table:

  ![image](https://github.com/shubhaygautam/NEXUS-A-Community-Detection-Platform/assets/111029251/647c864b-2dc6-41c7-a267-84d9e152d6cf)



- Visualize all communities with seed nodes:
 
  ![image](https://github.com/shubhaygautam/NEXUS-A-Community-Detection-Platform/assets/111029251/1d8c7775-0b7e-49e7-97b8-eaac19e19f28)




## Conclusion

This project develops a web-based platform to help big companies identify, engage, and collaborate with influencers within online communities. By leveraging advanced community detection and influence maximization algorithms, the platform provides actionable insights and comprehensive tools for effective influencer marketing strategies.
