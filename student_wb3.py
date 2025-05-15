
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        # Step 1: using  index to get the last element in the list

        my_index = len(self.open_list) - 1

        # Step 2: putting the selected candidate in that index in the variable next_soln
        next_soln = self.open_list[my_index]

        # Step 3:using pop to remove my_index
        self.open_list.pop(my_index)

        # Step 4: Returning the solution
        # <==== insert your pseudo-code and code above here
        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
           # Step 1: using 0 index to get the 1st element in the list
        my_index = 0

     # Step 2: putting the selected candidate in that index in the variable next_soln
        next_soln = self.open_list[my_index]

        # Step 3:  # Step 3:using pop to Remove the selected candidate from the open_list
        self.open_list.pop(my_index)

        # Step 4: Returning the solution

        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:
            return None

        # selectiing the best quality or the lowest value using quality  and putting it in findingbestchild
        findingBestChild = min(self.open_list, key=lambda x: x.quality)

        # removing the bestchild using remove because pop expects an index but remove removes the object directly 
        self.open_list.remove(findingBestChild)
        #assigning findingbestchild as next_soln
        next_soln = findingBestChild

        #  bestChild returns

        # <==== insert your pseudo-code and code above here
        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:  # doing like the psudocode stated
            return None
            # selects lowest value from the quality and variable value in findingbestchild
        findingbestChild = min(self.open_list, key=lambda x: x.quality + len(x.variable_values))

        # Removing  bestChild from open_list using remove as we aldready used pop
        self.open_list.remove(findingbestChild)
           #assigning findingbestchild as next_soln
        next_soln = findingbestChild

        #  bestChild returns

        # <==== insert your pseudo-code and code above here
        return next_soln
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
    #1st making the txt file as maze.txt
    maze = Maze(mazefile="maze.txt")  

    maze.contents[3][4] = hole_colour  # path tricking the dfs
    maze.contents[8][4] = wall_colour  # Blocking  DFS at the end

    maze.contents[10][6] = hole_colour  #  putting DFS trap
    maze.contents[14][6] = wall_colour  #Adding some Dead-end
    maze.contents[16][1] = hole_colour  # Dead-end
    maze.contents[19][4] = hole_colour  # Dead-end

    maze.contents[8][1] = hole_colour  #addition trap 
    maze.contents[12][9] = wall_colour #putting obtractle so that the dfs cannot enter
    maze.contents[11][12] = wall_colour  #blocking the other routes in the maze
    maze.contents[9][2] = wall_colour   #narrowing the path for dfs
    maze.contents[10][19] = wall_colour #putting end route barrier so that system can backtrack
    maze.contents[18][5] = wall_colour  #final dead-end for hindering dfs process




    # saving the file as txt in maze.txt 
    maze.save_to_txt("maze-breaks-depth.txt")  




    # <==== insert your code above here

def create_maze_depth_better():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work

     #1st making the txt file as maze.txt
    maze = Maze(mazefile="maze.txt")
    maze.contents[1][8] = wall_colour  #blocks direct path in the start of the maze
    maze.contents[9][10] = wall_colour  #mid maze obstacle
    maze.contents[15][6] = wall_colour  #makes dfs to explore alternative routes
    maze.contents[13][2] = wall_colour  # adding complexity in the lower left portion of the maze
    maze.contents[12][13] = wall_colour  # adding complexity in the right portion of the maze
    maze.contents[2][13] = wall_colour  # making the maze such that dfs doesnot get acess to easy verticle path in the maze
     # saving the file as txt in maze.txt 
    maze.save_to_txt("maze-depth-better.txt")  
    # <==== insert your code above here
