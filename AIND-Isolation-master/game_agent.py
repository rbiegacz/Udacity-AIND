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


def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w_pos, h_pos = game.width / 2., game.height / 2.
    y_pos, x_pos = game.get_player_location(player)
    return float((h_pos - y_pos)**2 + (w_pos - x_pos)**2)


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
    # TTODO: finish this function!
    # let's use the first function as
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score1 = len(game.get_legal_moves(player))
    score2 = len(game.get_legal_moves(game.get_opponent(player)))
    printd("The score is: {}".format(score1 - score2))
    return float(score1 - score2)


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
    # TTODO: finish this function!
    player_1_moves = game.get_legal_moves(player)
    player_2_moves = game.get_legal_moves(game.get_opponent(player))
    score1 = len(player_1_moves)
    score2 = len(player_2_moves)
    score3 = center_score(game, player)
    return float(score1 - score2 + score3)


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
    # TTODO: finish this function!
    score1 = len(game.get_legal_moves(player))
    score2 = len(game.get_legal_moves(game.get_opponent(player)))
    return float(score1 - 2*score2)


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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super(MinimaxPlayer, self).__init__(search_depth, score_fn, timeout)

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

    def active_player(self, game):
        """this function returns information whether we currently deal with active player"""
        return game.active_player == self

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
            _, best_move = self._minimax(game, depth, maximizing_player=self.active_player(game))
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
        # OK - we cannot go deepr
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
                #Do search until timeout
                printd("\n\nIterative Deepening - Round: {}".format(loop))
                best_move = self.alphabeta(game, loop)
                printd("Best move: {}".format(best_move))
                loop = loop + 1
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        # Return the best move from the last completed search iteration
        return best_move

    def active_player(self, game_state):
        """ Determination whether a given play is active"""
        return game_state.active_player == self

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

        if self.active_player(game):
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
            next_ply = game.forecast_move(move)
            score = self._alphabeta(next_ply, depth - 1, alpha, beta)[1]
            if maximizing:
                # if this is maximizer then let's decide if score is greater
                # than whatever we have seen so far
                if score > best_score:
                    best_score = score
                    best_move = move
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
                if score <= alpha:
                    best_move, best_score = move, score
                    break
                else:
                    beta = min(score, beta)
                    printd("beta: {}".format(beta))
        return best_move, best_score
