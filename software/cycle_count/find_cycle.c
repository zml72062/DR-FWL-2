#include <stdlib.h>
#include <string.h>

/**
 *  Linked list
 */
typedef struct List {
    long node;
    struct List* next;
} List;

/**
 *  Return an array of "list of neighbors" of every node, according to
 *  the given "edge_index" (in shape [2, num_of_edges]).
 * 
 *  @param edge_index the adjacency matrix, in shape [2, num_of_edges]
 *  @param num_nodes the number of nodes
 *  @param num_edges the number of edges
 *  
 *  @return an malloced array of "list of neighbors"
 */
static List** get_neighbor_list(long* edge_index, long num_nodes, long num_edges)
{
    List** neighbor_list = (List**)malloc(sizeof(List*) * num_nodes);
    for (long i = 0; i < num_nodes; i++)
    {
        List* ptr = (neighbor_list[i] = (List*)malloc(sizeof(List)));
        /**
         *  "ptr" is the current linked-list block awaiting to write
         */
        for (long j = 0; j < num_edges; j++)
        {
            if (edge_index[j] == i) /* edge start from i */
            {
                /* allocate the next linked-list block */
                ptr->next = (List*)malloc(sizeof(List));
                
                /* record the end point of the edge */
                ptr->node = edge_index[j + num_edges];

                ptr = ptr->next;
            }
        }
        ptr->node = -1; /* mark the last node of the linked list */
        ptr->next = NULL;
    }
    return neighbor_list;
}

/**
 *  Free the array of "list of neighbors" constructed by "get_neighbor_list"
 *  
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param num_nodes the number of nodes
 */
static void free_neighbor_list(List** neighbor_list, long num_nodes)
{
    for (long i = 0; i < num_nodes; i++)
    {
        /* Free linked list at "neighbor_list[i]" */
        List *ptr = neighbor_list[i], *ptr2 = ptr->next;
        while (ptr2 != NULL)
        {
            free(ptr);
            ptr = ptr2;
            ptr2 = ptr->next;
        }
        free(ptr);
    }
    free(neighbor_list);
}

/**
 *  Check if node j is a neighbor of node i.
 * 
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param i the first node
 *  @param j the second node
 *  
 *  @return whether node j is a neighbor of node i
 */
static int is_neighbor_of(List** neighbor_list, long i, long j)
{
    List* ptr = neighbor_list[i];
    while (ptr->node != -1)
    {
        if (ptr->node == j)
        {
            return 1;
        }
        ptr = ptr->next;
    }
    return 0;
}

/**
 *  Count paths of given length k that start at node i and end at node j.
 *
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param k the path length
 *  @param start the starting node
 *  @param end the ending node
 *  @param have_passed a vector to record whether a node has been visited,
 *  initially all 0 with length "num_nodes"
 * 
 *  @return number of length-k paths that start from "start" and end at "end"
 */
static long count_paths(List** neighbor_list, long k, long start, long end,
                        int* have_passed)
{
    have_passed[start] = 1;
    have_passed[end] = 1;

    if (k == 1)
    {
        return is_neighbor_of(neighbor_list, start, end);
    }

    long count = 0;
    for (List* ptr = neighbor_list[start]; ptr->node != -1; ptr = ptr->next)
    {
        long try_node = ptr->node;
        if (have_passed[try_node] == 1)
            continue;
        
        count += count_paths(neighbor_list, k - 1, try_node, end,
                             have_passed);
        have_passed[try_node] = 0;
    }
    return count;
}

/**
 *  Count cycles of given length k that pass a node i.
 *  
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param k the cycle length
 *  @param node the starting node
 *  @param num_nodes number of nodes
 *  
 *  @return number of k-cycles that pass node "node" 
 */
static long count_cycles(List** neighbor_list, long k, long node, long num_nodes)
{
    long count = 0;
    for (List* ptr = neighbor_list[node]; ptr->node != -1; ptr = ptr->next)
    {
        int* have_passed = (int*)calloc(num_nodes, sizeof(int));
        count += count_paths(neighbor_list, k - 1, node, ptr->node, 
                             have_passed);
        free(have_passed);
    }
    return count / 2;
}

/**
 *  Count paths between all pairs of nodes.
 *  
 *  @param edge_index the adjacency matrix, in shape [2, num_of_edges]
 *  @param num_nodes the number of nodes
 *  @param num_edges the number of edges
 *  @param k the path length
 *  @param paths_matrix the matrix in which to store the path counts, in
 *  shape [num_nodes, num_nodes]
 */
void graph_count_paths(long* edge_index, long num_nodes, long num_edges, long k,
                       long* paths_matrix)
{
    List** neighbor_list = get_neighbor_list(edge_index, num_nodes, num_edges);
    for (long i = 0; i < num_nodes; i++)
    {
        for (long j = 0; j < num_nodes; j++)
        {
            if (j == i)
                continue;
            int* have_passed = (int*)calloc(num_nodes, sizeof(int));
            paths_matrix[i * num_nodes + j] = count_paths(neighbor_list, k, i, j, 
                                                          have_passed);
            free(have_passed);
        }
    }
    free_neighbor_list(neighbor_list, num_nodes);
}

/**
 *  Node-level count cycles for every node in a graph.
 * 
 *  @param edge_index the adjacency matrix, in shape [2, num_of_edges]
 *  @param num_nodes the number of nodes
 *  @param num_edges the number of edges
 *  @param k the cycle length
 *  @param cycle_list the array in which to store the cycle counts, in
 *  shape [num_nodes, ]
 */
void graph_count_cycles(long* edge_index, long num_nodes, long num_edges, long k,
                        long* cycle_list)
{
    List** neighbor_list = get_neighbor_list(edge_index, num_nodes, num_edges);
    for (long i = 0; i < num_nodes; i++)
    {
        cycle_list[i] = count_cycles(neighbor_list, k, i, num_nodes);
    }
    free_neighbor_list(neighbor_list, num_nodes);
}

/**
 *  Node-level count 4-cliques for a given node i.
 * 
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param node the node i
 *  @param num_nodes number of nodes (for convenience we add this argument)
 * 
 *  @return number of 4-cliques that pass node i
 */
static long count_4_cliques(List** neighbor_list, long node, long num_nodes)
{
    long count = 0;
    for (List* n1 = neighbor_list[node]; n1->node != -1; n1 = n1->next)
    {
        for (List* n2 = n1->next; n2->node != -1; n2 = n2->next)
        {
            for (List* n3 = n2->next; n3->node != -1; n3 = n3->next)
            {
                if (
                    is_neighbor_of(neighbor_list, n1->node, n2->node)
                 && is_neighbor_of(neighbor_list, n2->node, n3->node)
                 && is_neighbor_of(neighbor_list, n3->node, n1->node)
                )
                    count++;
            }
        }
    }
    return count;
}

/**
 *  Node-level count chordal-cycles for a given node i.
 * 
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param node the node i
 *  @param num_nodes number of nodes
 * 
 *  @return number of chordal-cycles that pass node i
 */
static long count_chordal_cycles(List** neighbor_list, long node, long num_nodes)
{
    long count = 0;
    for (List* n1 = neighbor_list[node]; n1->node != -1; n1 = n1->next)
    {
        for (List* n2 = n1->next; n2->node != -1; n2 = n2->next)
        {
            if (is_neighbor_of(neighbor_list, n1->node, n2->node))
            {
                int* have_passed = (int*)calloc(num_nodes, sizeof(int));
                have_passed[node] = 1;
                count += count_paths(neighbor_list, 2, n1->node, n2->node, have_passed);
                free(have_passed);
            }
        }
    }
    return count;
}

/**
 *  Node-level count triangle-rectangles for a given node i.
 * 
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param node the node i
 *  @param num_nodes number of nodes
 * 
 *  @return number of triangle-rectangles that pass node i
 */
static long count_triangle_rectangles(List** neighbor_list, long node, long num_nodes)
{
    long count = 0;
    for (List* n1 = neighbor_list[node]; n1->node != -1; n1 = n1->next)
    {
        for (List* n2 = n1->next; n2->node != -1; n2 = n2->next)
        {
            if (is_neighbor_of(neighbor_list, n1->node, n2->node))
            {
                int* have_passed = (int*)calloc(num_nodes, sizeof(int));
                have_passed[node] = 1;
                count += count_paths(neighbor_list, 3, n1->node, n2->node, have_passed);
                free(have_passed);
            }
        }
    }
    return count;
}

/**
 * Node-level count tailed-triangles for a given node i.
 * 
 *  @param neighbor_list the malloced array of "list of neighbors"
 *  @param node the node i
 *  @param num_nodes number of nodes
 * 
 *  @return number of tailed-triangles that pass node i
 */
static long count_tailed_triangles(List** neighbor_list, long node, long num_nodes)
{
    long count = 0;
    for (List* n = neighbor_list[node]; n->node != -1; n = n->next)
    {
        int* have_passed = (int*)calloc(num_nodes, sizeof(int));
        long edge_triangle = count_paths(neighbor_list, 2, node, n->node, have_passed);
        free(have_passed);
        count += (count_cycles(neighbor_list, 3, n->node, num_nodes) - edge_triangle);
    }
    return count;
}

/**
 *  Node-level count 4-cliques, chordal-cycles, or triangle-rectangles 
 *  for every node in a graph.
 * 
 *  @param edge_index the adjacency matrix, in shape [2, num_of_edges]
 *  @param num_nodes the number of nodes
 *  @param num_edges the number of edges
 *  @param list the array in which to store the substructure counts, in
 *  shape [num_nodes, ]
 *  @param query the substructure to count
 */
void graph_count_substruct(long* edge_index, long num_nodes, long num_edges,
                           long* list, char* query)
{
    List** neighbor_list = get_neighbor_list(edge_index, num_nodes, num_edges);
    if (query[0] == 't' && query[1] == 'r') /* triangle-rectangles */
    {
        for (long i = 0; i < num_nodes; i++)
        {
            list[i] = count_triangle_rectangles(neighbor_list, i, num_nodes);
        }
    }
    else if (query[0] == 'c' && query[1] == 'l') /* cliques */
    {
        for (long i = 0; i < num_nodes; i++)
        {
            list[i] = count_4_cliques(neighbor_list, i, num_nodes);
        }
    }
    else if (query[0] == 'c' && query[1] == 'c') /* chordal-cycles */
    {
        for (long i = 0; i < num_nodes; i++)
        {
            list[i] = count_chordal_cycles(neighbor_list, i, num_nodes);
        }
    }
    else if (query[0] == 't' && query[1] == 't') /* tailed-triangles */
    {
        for (long i = 0; i < num_nodes; i++)
        {
            list[i] = count_tailed_triangles(neighbor_list, i, num_nodes);
        }
    }
    free_neighbor_list(neighbor_list, num_nodes);
}