from typing import List, Callable, Set
import re

WHITE = "white"
BLACK = "black"

class Evaluator:

    def __init__(self, eval_fns: List[Callable[[str, str], float]]):
        self.eval_fns = eval_fns

    @staticmethod
    def preprocess_string(s: str) -> str:
        """
        Preprocess the string.
        """
        return s.strip().rstrip(".").lower()

    def evaluate(self, pred: str, ref: str) -> dict:
        """
        Evaluate the predicted and reference strings.
        """
        pred = self.preprocess_string(pred)
        ref = self.preprocess_string(ref)
        scores = {}
        for eval_fn in self.eval_fns:
            score = eval_fn(pred, ref)
            key = eval_fn.__name__
            scores[key] = score
        return scores
    
    def empty_results(self) -> dict:
        """
        Return an empty results dictionary.
        """
        results = {}
        for eval_fn in self.eval_fns:
            key = eval_fn.__name__
            results[key] = []
        return results

class PuzzleEvaluator(Evaluator):

    @staticmethod
    def eval_puzzles_max_matching(pred: str, ref: str) -> float:
        """
        Compute the maximum substring match between the predicted and reference strings
        from the beginning of the strings.
        """
        denom = len(ref)
        max_substring = 0
        for i in range(min(len(pred), len(ref))):
            if pred[i] == ref[i]:
                max_substring += 1
            else:
                break
        return max_substring / denom

    @staticmethod
    def eval_puzzles_first_move(pred: str, ref: str) -> int:
        """
        Compute the match between the first word of the predicted and reference strings.
        """
        delim = ","
        pred = pred.lower().split(delim)[0]
        ref = ref.lower().split(delim)[0]
        return int(pred == ref)

    def __init__(self):
        super().__init__([self.eval_puzzles_max_matching, self.eval_puzzles_first_move])

class PuzzleWithThemesEvaluator(Evaluator):

    @staticmethod
    def isolate_moves(s: str) -> List[str]:
        """
        Isolate the moves from the string.
        """
        tgt = "sequence is"
        output = s.strip().split(tgt)[-1].strip()
        return output

    @staticmethod
    def eval_pwt_puzzles_max_matching(pred: str, ref: str) -> float:
        """
        Compute the maximum substring match between the predicted and reference strings
        from the beginning of the strings.
        """
        pred = PuzzleWithThemesEvaluator.isolate_moves(pred)
        ref = PuzzleWithThemesEvaluator.isolate_moves(ref)
        denom = len(ref)
        max_substring = 0
        for i in range(min(len(pred), len(ref))):
            if pred[i] == ref[i]:
                max_substring += 1
            else:
                break
        return max_substring / denom

    @staticmethod
    def eval_pwt_puzzles_first_move(pred: str, ref: str) -> int:
        """
        Compute the match between the first word of the predicted and reference strings.
        """
        pred = PuzzleWithThemesEvaluator.isolate_moves(pred)
        ref = PuzzleWithThemesEvaluator.isolate_moves(ref)
        delim = ","
        pred = pred.lower().split(delim)[0].strip()
        ref = ref.lower().split(delim)[0].strip()
        return int(pred == ref)

    def __init__(self):
        super().__init__([self.eval_pwt_puzzles_max_matching, self.eval_pwt_puzzles_first_move])

class CheckmateInOneEvaluator(Evaluator):

    @staticmethod
    def eval_checkmate_in_one_accuracy(pred: str, ref: str) -> int:
        """
        Compute the accuracy of the predicted checkmate in one move.
        """
        return int(ref.rstrip("#") in pred)

    def __init__(self):
        super().__init__([self.eval_checkmate_in_one_accuracy])

class ChessStateTrackingEvaluator(Evaluator):

    @staticmethod
    def str_to_moves_set(s: str) -> Set:
        """
        Convert the moves to a list.
        """
        delim = "\\"
        return set(s.strip().split(delim))

    @staticmethod
    def eval_chess_state_tracking(pred: str, ref: str) -> float:
        """
        Compute the accuracy of the predicted checkmate in one move.
        """
        pred_moves = ChessStateTrackingEvaluator.str_to_moves_set(pred)
        ref_moves = ChessStateTrackingEvaluator.str_to_moves_set(ref)
        iou = len(pred_moves.intersection(ref_moves)) / len(pred_moves.union(ref_moves))
        return iou

    def __init__(self):
        super().__init__([self.eval_chess_state_tracking])

class ChessStateValueEvaluator(Evaluator):

    ADVANTAGE = "advantage"
    EQUAL = "equal"

    @staticmethod
    def eval_chess_state_value(pred: str, ref: str) -> int:
        """
        Assumptions on reference strings:
        There are only three types of reference strings:
        1. "black has advantage"
        2. "white has advantage"
        3. "the game is equal"
        """
        if ChessStateValueEvaluator.ADVANTAGE in ref:
            if WHITE in ref:
                return int(WHITE.lower() in pred
                           and ChessStateValueEvaluator.ADVANTAGE.lower() in pred)
            elif BLACK in ref:
                return int(BLACK.lower() in pred
                           and ChessStateValueEvaluator.ADVANTAGE.lower() in pred)
            else:
                return 0
        elif ChessStateValueEvaluator.EQUAL in ref:
            return int(ChessStateValueEvaluator.EQUAL in pred)
        else:
            return 0

    def __init__(self):
        super().__init__([self.eval_chess_state_value])

class ChessAnnotationMCQEvaluator(Evaluator):

    THRESH = 0.9

    @staticmethod
    def option_matching(pred: str, ref: str) -> int:
        """
        Compute the maximum substring match between the predicted and reference strings
        from any part of the strings to determine if selected option is correct.
        """
        m, n = len(pred), len(ref)
        # Create a 2D DP table with (m+1) x (n+1) dimensions
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        max_len = 0  # length of the longest common substring found so far

        # Build the table in bottom-up fashion.
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                else:
                    dp[i][j] = 0  # Reset because substrings must be contiguous

        # Return 1 if the longest common substring is at least 90% of the reference string
        return int(max_len / len(ref) > ChessAnnotationMCQEvaluator.THRESH)
    
    def __init__(self):
        super().__init__([self.option_matching])

class CombinedEvaluator(Evaluator):

    def __init__(self):
        super().__init__(
            [
                PuzzleEvaluator.eval_puzzles_max_matching,
                PuzzleEvaluator.eval_puzzles_first_move,
                PuzzleWithThemesEvaluator.eval_pwt_puzzles_max_matching,
                PuzzleWithThemesEvaluator.eval_pwt_puzzles_first_move,
                CheckmateInOneEvaluator.eval_checkmate_in_one_accuracy,
                ChessStateTrackingEvaluator.eval_chess_state_tracking,
                ChessStateValueEvaluator.eval_chess_state_value,
                ChessAnnotationMCQEvaluator.option_matching
            ]
        )

    @staticmethod
    def has_single_san(s: str) -> bool:
        pattern = r'^[a-h][1-8][a-h][1-8]$'
        return re.match(pattern, s)
    
    @staticmethod
    def start_with_option(s: str) -> bool:
        # Check if the string starts with an option number [integer followed by a period]
        pattern = r'^\d+\.\s'
        return re.match(pattern, s)

    def evaluate(self, pred: str, ref: str) -> dict:
        """
        Compute the combined evaluation score.
        """
        pred = self.preprocess_string(pred)
        ref = self.preprocess_string(ref)

        if self.start_with_option(ref):
            eval_class = ChessAnnotationMCQEvaluator
        elif "\\" in ref or self.has_single_san(ref):
            eval_class = ChessStateTrackingEvaluator
        elif "sequence is" in ref and "theme" in ref:
            eval_class = PuzzleWithThemesEvaluator
        elif "sequence is" in ref:
            eval_class = PuzzleEvaluator
        elif "#" in ref:
            eval_class = CheckmateInOneEvaluator
        elif ChessStateValueEvaluator.ADVANTAGE in ref or ChessStateValueEvaluator.EQUAL in ref:
            eval_class = ChessStateValueEvaluator
        else:
            print(f"Incorrect reference string: {ref}")
            return {}
            # raise ValueError("Invalid reference string")
        evaluator: Evaluator = eval_class()
        return evaluator.evaluate(pred, ref)

if __name__ == "__main__":
    ref = "3. The White queen eyes Black's weak e6-pawn for the final assault on the uncastled king."
    pred = "This is the answer: 3. The White queen eyes Black's weak e6-pawn for the final assault on the uncastled king."
    eval_combined = CombinedEvaluator()
    score = eval_combined.evaluate(pred, ref)
    print(score)
