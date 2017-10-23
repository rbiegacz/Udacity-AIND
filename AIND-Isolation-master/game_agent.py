"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


DEBUG = False


def printd(msg):
    """Printing debugging information"""
    if DEBUG:
        print(msg)


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def active_player(game, player):
    """this function returns information whether we currently deal with active player"""
    return game.active_player == player


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # let's use the first function as
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score1 = len(game.get_legal_moves(player))
    score2 = len(game.get_legal_moves(game.get_opponent(player)))
    # let's normalize the score
    # the max value of score1 might be 8
    # the max value of score2 might be 8
    # so the score can have vaulues from -8 up to 8
    score = float(score1 - score2)/8.0
    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location = (mid_w , mid_h)
    # getting player #1 location
    player_location  = game.get_player_location(player)
    # checking if player is the center location
    if center_location == player_location:
        # OK - we are strongly recommending this move
        score = score+1
    printd("The score for move {} is: {}".format(game.get_player_location(player), score))
    return float(score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")

    #player_1_moves = game.get_legal_moves(player)
    #player_2_moves = game.get_legal_moves(game.get_opponent(player))
    #score1 = len(player_1_moves)
    #score2 = len(player_2_moves)
    #score = float(score1 - score2)/8.0
    center_y_pos, center_x_pos = int(game.height/2), int(game.width/2)
    player_y_pos, player_x_pos = game.get_player_location(player)
    # mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    # center_location = (mid_w , mid_h)

    opponent_y_pos, opponent_x_pos = game.get_player_location(game.get_opponent(player))
    player_distance = abs(player_y_pos - center_y_pos) + abs(player_x_pos - center_x_pos)
    opponent_distance = abs(opponent_y_pos - center_y_pos) + abs(opponent_x_pos - center_x_pos)
    score = float(abs(opponent_distance - player_distance)/9)
    if player_distance < opponent_distance:
        pass
    elif player_distance > opponent_distance:
        score = -score

    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location = (mid_w , mid_h)
    # getting player #1 location
    player_location  = game.get_player_location(player)
    # checking if player is the center location
    if center_location == player_location:
        # OK - we are strongly recommending this move
        score = score+1
    printd("The score is: {}".format(score))
    return float(score)


def _custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")

    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location = (mid_w , mid_h)
    # getting players location
    player_location  = game.get_player_location(player)
    # checking if player is the center location
    if center_location == player_location:
        # returning heuristic1 with incentive
        score = custom_score(game, player)+100
    else:
        # returning heuristic1
        score = custom_score(game, player)
    printd("The score is: {}".format(score))
    return float(score)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")

    score1 = len(game.get_legal_moves(player))
    score2 = len(game.get_legal_moves(game.get_opponent(player)))
    # let's normalize the score
    # the max value of score1 might be 8
    # the max value of 2*score2 might be 16
    # so the score can have values from -16 up to 8
    score = float(score1 - 2*score2)/16.0
    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location = (mid_w , mid_h)
    # getting player #1 location
    player_location  = game.get_player_location(player)
    # checking if player is the center location
    if center_location == player_location:
        # OK - we are strongly recommending this move
        score = score+1
    printd("The score for move {} is: {}".format(game.get_player_location(player), score))
    return score


class IsolationPlayer(object):
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=40.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)

        # if this is the first move let's pick the center of the board
        if game.move_count == 0:
            return int(game.height/2), int(game.width/2)

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TTODO: finish this function!
        best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            _, best_move = self._minimax(game, depth, maximizing_player=active_player(game, self))
            printd("Best move: {}".format(best_move))
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        return best_move

    def _minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move_so_far = (-1, -1)
        lowest_score_so_far = float("inf")
        highest_score_so_far = float("-inf")
        # OK - we cannot go deeper
        # just report score for current player's position
        if depth == 0:
            score = self.score(game, self)
            printd("Depth: 0, Score: {}\n".format(score))
            return score, best_move_so_far

        legal_moves = game.get_legal_moves()
        # if there are no moves any more - report it
        if not legal_moves:
            if maximizing_player:
                return highest_score_so_far, best_move_so_far
            else:
                return lowest_score_so_far, best_move_so_far

        if maximizing_player:
            for move in legal_moves:
                # Evaluate this move in depth.
                score, _ = self._minimax(game.forecast_move(move), depth-1, maximizing_player=False)
                # If this branch yields a sure win, no need to search further.
                # Otherwise, remember the best move.
                if score == float("inf"):
                    highest_score_so_far, best_move_so_far = score, move
                    break
                if score > highest_score_so_far:
                    highest_score_so_far, best_move_so_far = score, move
            return highest_score_so_far, best_move_so_far
        else:
            for move in legal_moves:
                # Evaluate this move in depth.
                score, _ = self._minimax(game.forecast_move(move), depth-1, maximizing_player=True)
                # If this branch yields a sure win, no need to search further.
                # Otherwise, remember the best move.
                if score == float("-inf"):
                    lowest_score_so_far, best_move_so_far = score, move
                    break
                if score < lowest_score_so_far:
                    lowest_score_so_far, best_move_so_far = score, move
            return lowest_score_so_far, best_move_so_far


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
        """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TTODO: finish this function!
        best_move = (-1, -1)

        # OK - if there are no moved to do we return (-1, -1)
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return best_move

        # OK - if this is beginning of the play we return board central point
        if game.move_count == 0:
            return int(game.height/2), int(game.width/2)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            loop = 1
            while 1:
            #while loop <= self.search_depth:
                #Do search until timeout
                printd("\n\nIterative Deepening - Round: {}".format(loop))
                best_move = self.alphabeta(game, loop)
                printd("Best move: {}".format(best_move))
                loop = loop + 1
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return self._alphabeta(game, depth, alpha, beta)[0]

    def _alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """
        again this just wraps the old code logic that returned a tuple of (move, score)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)

        # depth 0 means that we really are asked just to score the game
        if depth == 0:
            score = self.score(game, self)
            printd("Depth: 0, Score: {}\n".format(score))
            return best_move, score

        if active_player(game, self):
            # OK - now we run for Maximizer
            best_score = float("-inf")
            maximizing = True
            printd("\nMaximizing player...")
        else:
            # OK - now we run for Minimizer
            best_score = float("inf")
            printd("\nMinimizing player...")
            maximizing = False

        legal_moves = game.get_legal_moves()
        printd("Alphabeta - depth: {}, alpha: {}, beta {}".format(depth, alpha, beta))
        printd("Legal moves: {}".format(legal_moves))
        for move in legal_moves:
            printd("Selecting move: {}".format(move))
            score = self._alphabeta(game.forecast_move(move), depth - 1, alpha, beta)[1]
            printd("Alphabeta - depth: {}, alpha: {}, beta {}".format(depth, alpha, beta))
            printd("Selecting move: {} which has a score {}".format(move, score))
            if maximizing:
                # if this is maximizer then let's decide if score is greater
                # than whatever we have seen so far
                if score > best_score:
                    best_score = score
                    best_move = move
                elif score == best_score:
                    printd("Score {} achieved also for {} move".format(score, move))
                if best_score >= beta:
                    best_move, best_score = move, score
                    break
                else:
                    alpha = max(score, alpha)
                    printd("alpha: {}".format(alpha))
            else:
                # if this is minimizer then let's decide if score is smaller
                # than whatever we have seen so far
                if score < best_score:
                    best_score = score
                    best_move = move
                elif score == best_score:
                    printd("Score {} achieved also for {} move".format(score, move))
                if score <= alpha:
                    best_move, best_score = move, score
                    break
                else:
                    beta = min(score, beta)
                    printd("beta: {}".format(beta))
        return best_move, best_score
