#include <iostream>
#include <functional>
#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <thread>
#include "NN.hpp"

// fix the errors between the NN.hpp and traning.cpp

// Macro definitions
#define BOARD_SIZE 19
const int PASS_MOVE = BOARD_SIZE * BOARD_SIZE;

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
    board->scoreMargin = 0.0;
    board->player = 0;
}
static void floodFillGroup(const Board &board, int x, int y, CellState color,
                           std::vector<std::pair<int,int>>& stones,
                           std::vector<std::pair<int,int>>& liberties,
                           bool visited[BOARD_SIZE][BOARD_SIZE]) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return;
    if (visited[x][y] || board.cells[x][y] != color) return;

    visited[x][y] = true;
    stones.push_back({x, y});

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
            if (board.cells[nx][ny] == EMPTY) {
                // Check if we already counted this liberty to avoid double-counting
                if (std::find(liberties.begin(), liberties.end(), std::make_pair(nx, ny)) == liberties.end()) {
                    liberties.push_back({nx, ny});
                }
            }
        }
    }

    floodFillGroup(board, x + 1, y, color, stones, liberties, visited);
    floodFillGroup(board, x - 1, y, color, stones, liberties, visited);
    floodFillGroup(board, x, y + 1, color, stones, liberties, visited);
    floodFillGroup(board, x, y - 1, color, stones, liberties, visited);
}
// converting the board to NN input
std::vector<std::vector<std::vector<double>>> boardToNNInput(const Board &board, CellState perspective) {
    // the 3 states: empty, own stone, opponent's stone
    std::vector<std::vector<std::vector<double>>> input(5, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0)));

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

    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if(board.cells[r][c] != EMPTY && !visited[r][c]) {
                std::vector<std::pair<int, int>> stones;
                std::vector<std::pair<int, int>> libs;
                CellState group = board.cells[r][c];

                floodFillGroup(board, r, c, group, stones, libs, visited);

                int libertiesCount = libs.size();

                double heatIntensity = (libertiesCount > 0) ? (1.0 / static_cast<double>(libertiesCount)) : 0.0;

                int targetChannel = (group == perspective) ? 3 : 4;

                for (const auto &stone : stones) {
                    input[targetChannel][stone.first][stone.second] = heatIntensity;
                }
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


bool makeMove(Board *board, int x, int y, int &out_captures, int &out_connections) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board->cells[x][y] != EMPTY) return false;

    CellState player   = board->turn;
    CellState opponent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    // 1. Place the stone FIRST
    board->cells[x][y] = player;

    // 2. Then resolve captures (now the enemy group may be short a liberty)
    int capts = 0;
    capts += removeGroup(board, x + 1, y, opponent);
    capts += removeGroup(board, x - 1, y, opponent);
    capts += removeGroup(board, x, y + 1, opponent);
    capts += removeGroup(board, x, y - 1, opponent);

    // 3. Suicide check last. If we captured anything we necessarily have a liberty.
    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if (capts == 0 && !groupLiberties(board, x, y, player, visited)) {
        board->cells[x][y] = EMPTY;   // illegal, roll back
        return false;
    }

    out_captures = capts;

    int conns = 0;
    int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board->cells[nx][ny] == player)
            conns++;
    }
    out_connections = conns;

    if (player == BLACK_STONE) board->score[0] += capts;
    else                       board->score[1] += capts;

    board->lastMove[0] = x;
    board->lastMove[1] = y;
    board->turn = opponent;
    board->passes = 0;
    return true;
}

// finds a legal action to sample from
int sampleAction(const Board* board, const std::vector<double>& probs) {
    std::vector<double> legal_probs(probs.size(), 0.0);
    legal_probs[PASS_MOVE] = probs[PASS_MOVE];
    if(legal_probs[PASS_MOVE] == 0.0){
        legal_probs[PASS_MOVE] = 1.0;
    }
    bool has_legal = false;
    double total_prob = 0.0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        int r = i / BOARD_SIZE;
        int c = i % BOARD_SIZE;
        if (board->cells[r][c] == EMPTY) {
            legal_probs[i] = probs[i];
            total_prob += probs[i];
            has_legal = true;
        }
    }
    total_prob += probs[PASS_MOVE];

    if (total_prob <= 0.0){
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;
            legal_probs[i] = (board->cells[r][c] == EMPTY) ? 1.0 : 0.0;
        }
        total_prob = static_cast<double>(BOARD_SIZE * BOARD_SIZE);
    }

    static thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(legal_probs.begin(), legal_probs.end());
    return dist(gen);
}
// scoring & territory tracking
CellState winner(Board &board) {
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
                // finding the territory with flood fill of each player for scoring.
                // this should be add to the frontend for scoring at the end of the game.
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
                    scoreW += territory;
                } else if (touchesBlack && !touchesWhite) {
                    scoreB += territory;
                }
            }
        }
    }
    board.score[0] = scoreB;
    board.score[1] = scoreW;

    const double KOMI = 6.5;
    double finalB = scoreB;
    double finalW = scoreW;
    if (finalB - finalW == 0.0) finalW += KOMI;
    board.scoreMargin = finalB - finalW;
    if (finalB > finalW) {
        std::cout << "Black wins!" << std::endl;
        board.winner = false;
        return BLACK_STONE; // Correctly returns CellState instead of bool
    } else {
        std::cout << "White wins!" << std::endl;
        board.winner = true;
        return WHITE_STONE; // Correctly returns CellState instead of bool
    }
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
        int failed = 0;
        while (turns < maxTurns) {
            CellState activePlayer = board.turn;
            auto inputs = boardToNNInput(board, activePlayer);
            ai.forwardPropagate(inputs);
        
            int move = sampleAction(&board, ai.probs);
        
            if (move == PASS_MOVE) {
                board.passes++;
                board.turn = (activePlayer == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                if (board.passes >= 2) break;
                continue;
            }
        
            int x = move / BOARD_SIZE, y = move % BOARD_SIZE;
            int captures = 0, connections = 0;
        
            if (makeMove(&board, x, y, captures, connections)) {
                failed = 0;
                if (!memsteps.empty() && memsteps.back().player != activePlayer)
                    memsteps.back().stonesLost = captures;
                memsteps.push_back({std::move(inputs), move, activePlayer,
                                    captures, 0, connections, {}});
                turns++;
            } else if (++failed > 10) {
                board.passes++;
                board.turn = (activePlayer == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                failed = 0;
                if (board.passes >= 2) break;
            }
        }

        CellState winningPlayer = winner(board);
        ai.trainOnEpisodes(memsteps, winningPlayer, board.scoreMargin);

        std::cout << "[Game" << i << "] -> Final Score|" << "turns: " << turns << " | Black: " << board.score[0] << " | White: " << board.score[1] << " | Margin: " << board.scoreMargin << " | Winner: " << (winningPlayer == BLACK_STONE ? "Black" : "White") << std::endl;
        std::cout << board.scoreMargin << std::endl;
        if(i % 10 == 0) {
            std::cout << "Finished game " << i << " | Winner: " << (winningPlayer == BLACK_STONE ? "Black" : "White") << " | File Checkpoint Saved." << std::endl;
            ai.save("model_weights.txt");
        }
    }
    return 0;
}
