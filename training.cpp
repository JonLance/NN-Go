
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "NN.hpp"
// Macro definitions
#define BOARD_SIZE 19
#define CELL_SIZE 36
#define MARGIN 30
#define WINDOW_SIZE (CELL_SIZE * (BOARD_SIZE - 1) + 2  * MARGIN)

// uses cell state enum to determine what color to draw for each cell and the turn of the player
// Enum for cell states
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
};



struct exp{
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

int main() {
    const int INPUT_SIZE = 1083; // 19x19x3
    const int OUTPUT_SIZE = 361;
    const auto HIDDEN_LAYERS = 256;

    double gamma = 0.99;
    double eps = 1.0;
    double epsdecay = 0.995;
    double minEps = 0.01;
    double batchSize = 32;
    NNGo nn(INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE);
    
    for (int runs = 0; runs < 5000; ++runs){
        train();
        
    }
    return 0;
}
