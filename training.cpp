#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include "NN.hpp"

// still needs improvement

// Macro definitions
#define BOARD_SIZE 19


enum CellState {
    EMPTY = 0,
    BLACK_STONE = 1,
    WHITE_STONE = 2,
};
struct Board {
    CellState cells[BOARD_SIZE][BOARD_SIZE];
    CellState turn;
    int score[2];
    int lastMove[2];
    int passes;
    bool winner; // let true be black win, false be white win

};

struct memstep {
    std::vector<std::vector<std::vector<double>>> state;
    int action;
    CellState player;

};

void  initBoard(Board *board) {
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board->cells[r][c] = EMPTY;
        }
    }
    board->turn = BLACK_STONE;
    board->score[0] = 0; // black
    board->score[1] = 0; // white
    board->lastMove[0] = -1;
    board->lastMove[1] = -1;
    board->passes = 0;
    board->winner = false;
}
// converting the board to NN input
std::vector<std::vector<std::vector<double>>> boardToNNInput(const Board &board, CellState perspective) {
    // the 3 states: empty, own stone, opponent's stone
    std::vector<std::vector<std::vector<double>>> input(3, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0)));

    CellState opponent = (perspective == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board.cells[r][c] == perspective) {
                input[0][r][c] = 1.0;
            } else if (board.cells[r][c] == opponent) {
                input[1][r][c] = 1.0;
            } else {
                input[2][r][c] = 1.0;
            }
        }
    }
    return input;
}

static bool groupLiberties(Board *board, int x, int y, CellState color, bool visited[BOARD_SIZE][BOARD_SIZE]) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (visited[x][y]) return false;
    if (board->cells[x][y] == EMPTY) return true;   // found a liberty
    if (board->cells[x][y] != color) return false;  // enemy stone, dead end

    visited[x][y] = true;
    return groupLiberties(board, x + 1, y, color, visited)
        || groupLiberties(board, x - 1, y, color, visited)
        || groupLiberties(board, x, y + 1, color, visited)
        || groupLiberties(board, x, y - 1, color, visited);
}
// remove the stones that are surrounded by the opponent's stones in a group
static int removeLiberties(Board *board, int x, int y, CellState color) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return 0;
    if (board->cells[x][y] != color) return 0;
    board->cells[x][y] = EMPTY;
    int n = 1;
    n += removeLiberties(board, x + 1, y, color);
    n += removeLiberties(board, x - 1, y, color);
    n += removeLiberties(board, x, y + 1, color);
    n += removeLiberties(board, x, y - 1, color);
    return n;
}

static int removeGroup(Board *board, int x, int y, CellState oppColor) {
    if(x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return 0;
    if(board->cells[x][y] != oppColor) return 0;
    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if(!groupLiberties(board, x, y, oppColor, visited)) {
        return removeLiberties(board, x, y, oppColor);
    }
    return 0;
}


bool makeMove(Board *board, int x, int y, CellState color) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board->cells[x][y] != EMPTY) return false;

    CellState player = board->turn;
    CellState oppent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    board->cells[x][y] = player;
    int capts = 0;
    capts += removeGroup(board, x + 1, y, oppent);
    capts += removeGroup(board, x - 1, y, oppent);
    capts += removeGroup(board, x, y + 1, oppent);
    capts += removeGroup(board, x, y - 1, oppent);


    if (player == BLACK_STONE) board->score[0] += capts;
    else board->score[1] += capts;

    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if(!groupLiberties(board, x, y, player, visited)) {
        board->cells[x][y] = EMPTY;
        return false;
    }
    // Switch to the opponent's turn
    board->turn = oppent;
    board->passes = 0;
    return true;
}


int Moves (const Board *board, const std::vector<double> &probs) {
    std::vector<double> probsOfIlligal(probs.size(), 0.0);
    int count = 0;

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        // g = x, f = y to not mess with the x and y of the boad
        int g = i / BOARD_SIZE;
        int f = i % BOARD_SIZE;
        if (board->cells[g][f] == EMPTY) {
            probsOfIlligal[i] = probs[i];
            count++;
        }
    }

    if (count <= 0){
        //can not move
        // might have to force a move if the it is deadlocked
        return -1; // should return pass
    }
    for(double &p : probsOfIlligal) {
        p /= count;
    }

    // distribution sample
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> dist(probsOfIlligal.begin(), probsOfIlligal.end());

    double r = dist(gen);
    double sample = 0.0;
    for (int i = 0; i < probsOfIlligal.size(); ++i) {
        sample += probsOfIlligal[i];
        if (sample <= r) return i;
    }
    return probsOfIlligal.size() - 1;
}
bool winner(Board &board, int Bcapts, int Wdeadstones, int Wcapts, int Bdeadstones) {
    /*
     * the winner is the player with the higher score
     * the score is calculated by
     * {
     *  territory = 1 point per connected empty cell
     *  captures = 1 point per captured opponent stone
     *  deadstones = 2 points per dead stone
     * }
     */

    int scoreW = board.score[1];
    int scoreB = board.score[0];
    // cordant traking for territory with flood fill
    std::vector<std::vector<bool>> visited(BOARD_SIZE, std::vector<bool>(BOARD_SIZE, false));
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board.cells[i][j] == EMPTY && !visited[i][j]) {
                int territory = 0;
                bool touchesBlack = false;
                bool touchesWhite = false;

                std::function<void(int, int)> dfs = [&](int x, int y) {
                    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || visited[x][y]) return;

                    if (board.cells[x][y] == BLACK_STONE) {
                        touchesBlack = true;
                        return;
                    }
                    if (board.cells[x][y] == WHITE_STONE) {
                        touchesWhite = true;
                        return;
                    }
                    if (visited[x][y]) return;
                    visited[x][y] = true;
                    territory++;
                    dfs(x + 1, y);
                    dfs(x - 1, y);
                    dfs(x, y + 1);
                    dfs(x, y - 1);
                };
                dfs(i, j);
                if (!touchesBlack && touchesWhite) {
                    scoreB += territory;
                } else if (touchesBlack && !touchesWhite) {
                    scoreW += territory;
                }
            }
        }
    }

    scoreB += Bcapts;
    scoreW += Wcapts;
    scoreB += Bdeadstones * 2;
    scoreW += Wdeadstones * 2;

    if (scoreB > scoreW) {
        std::cout << "Black wins!" << std::endl;
        board.winner = false;
    } else if (scoreB < scoreW) {
        std::cout << "White wins!" << std::endl;
        board.winner = true;
    } else {
        std::cout << "It's a tie!" << std::endl;
    }
    std::cout << "Black score: " << scoreB << ", White score: " << scoreW << std::endl;
    return board.winner;
}

int main() {
    NNGo ai;
    ai.initWeights();

    // loading the pre-existing brain if available
    if (ai.load("model_weights.txt")) {
        std::cout << "Loaded model weights from model_weights.txt..." << std::endl;
    } else {
        std::cout << "Starting fresh training..." << std::endl;
    }

    Board board;
    int maxGames = 5000; // number of games to train at a time
    for (int i = 0; i < maxGames; ++i) {
        initBoard(&board);
        std::vector<memstep> memsteps;
        int turns = 0;
        int maxTurns = 220; // maximum number of turns per game to avoid infinite loops
        while (turns < maxTurns) {
            CellState activePlayer = board.turn;
            auto inputs = boardToNNInput(board, activePlayer);
            // forward propagation to get predictions
            ai.forwardPropagate(inputs);
            // predictions and sampling actions
            int move = sampleFromDistribution(ai.probs, board);
            if (move == -1) {
                board.passes++;
                if (board.passes == 2) break;
                continue;
            }
            int x = move / BOARD_SIZE;
            int y = move % BOARD_SIZE;
            if (makeMove(&board, x, y, board.turn)) {
                memsteps.push_back({x, y, board.turn});
                turns++;
            }
            if (turns++) {
                board.passes = 0;
                board.turn = (board.turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
            }

        }

    }
}
