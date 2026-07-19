/*
 * input: 3 channels x 19 x 19  = 1083 values (flattened)
 * hidden: layer0  1083 -> 256
 *         layer1   256 -> 256
 *         layer2   256 -> 256
 * output:          256 -> 361 (softmax)
 *
 *should change the way hidden layers are made with changing hidden_layers between the hidden layers
 * this is still buggy not the finshed version
 */
#ifndef GO_BOARD_SIZE
#define GO_BOARD_SIZE 19
#endif

#ifndef NN_HPP
#define NN_HPP


#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <fstream>


enum CellState {
    EMPTY = 0,
    BLACK_STONE = 1,
    WHITE_STONE = 2,
};

struct Board {
    CellState cells[GO_BOARD_SIZE][GO_BOARD_SIZE];
    CellState turn;
    int score[2];
    int lastMove[2];
    int passes;
    double scoreMargin;
    int player;
    bool winner;
};
int pass = 0;


struct memstep {
    std::vector<std::vector<std::vector<double>>> state;
    int action;
    CellState player;
    int captsMade;
    int stonesLost;
    int connections;
};


class NNGo {
public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int INPUT_CHANNELS = 5;

    static constexpr int INPUT_SIZE = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE; // 1083
    static constexpr int OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE;                  // 361

    static constexpr int HIDDEN_LAYERS = 5;
    static constexpr int HIDDEN_NEURONS = 512;

    double learning_rate = 0.01;

    // State captured during the forward pass (needed by backprop).
    std::vector<double> input_flat;   // INPUT_SIZE
    std::vector<std::vector<double>> hidden;       // [layer][neuron]
    std::vector<double> logits;       // OUTPUT_SIZE (pre-softmax)
    std::vector<double> probs;        // OUTPUT_SIZE (post-softmax)

    // weights[l][neuron][input], biases[l][neuron].
    // l in [0, HIDDEN_LAYERS-1] are hidden layers; l == HIDDEN_LAYERS is output.
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    void initWeights();
    void forwardPropagate(const std::vector<std::vector<std::vector<double>>>& X);
    void backwardPropagate(const std::vector<double>& y_true, double reward);
    void train(const std::vector<std::vector<std::vector<std::vector<double>>>>& X,const std::vector<std::vector<double>>& y,int epochs);
    void trainOnEpisodes(const std::vector<memstep>& episodes, CellState winner, double finalMargin);
    double crossEntropyLoss(const std::vector<double>& y_true,const std::vector<double>& y_pred) const;
    bool save(const std::string& filename);
    bool load(const std::string& filename);

private:
    void softmax();
    int  layerInputSize(int l) const { return l == 0 ? INPUT_SIZE : HIDDEN_NEURONS; }
};
#endif
bool NNGo::save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file to save weights: " << filename << std::endl;
        return false;
    }

    out << HIDDEN_LAYERS << " " << HIDDEN_NEURONS << std::endl;

    for (int l = 0; l <= HIDDEN_LAYERS; ++l) {
        int rows = (l == HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_NEURONS;
        int cols = layerInputSize(l);

        // Save weights
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out << weights[l][i][j] << " ";
            }
        }

        // Save biases (Using 'rows' ensures it matches output layer size safely)
        for (int i = 0; i < rows; ++i) {
            out << biases[l][i] << " ";
        }
        out << std::endl;
    }

    out.close();
    return true;
}
static inline double relu(double x)            { return x > 0 ? x : 0.0; }
static inline double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
bool NNGo::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    int file_hidden_layers, file_hidden_neurons;
    if (!(in >> file_hidden_layers >> file_hidden_neurons)) {
        std::cout << "Failed to read hidden layers/neurons from file: " << filename << std::endl;
        return false;
    }
    if (file_hidden_layers != HIDDEN_LAYERS || file_hidden_neurons != HIDDEN_NEURONS) {
        std::cout << "Hidden layers/neurons mismatch: " << file_hidden_layers << "x" << file_hidden_neurons << " vs " << HIDDEN_LAYERS << "x" << HIDDEN_NEURONS << std::endl;
        return false;
    }

    weights.assign(HIDDEN_LAYERS + 1, {});
    biases.assign(HIDDEN_LAYERS + 1, {});

    for (int l = 0; l <= HIDDEN_LAYERS; ++l) {
        int rows = (l == HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_NEURONS;
        int cols = layerInputSize(l);
        weights[l].assign(rows, std::vector<double>(cols));
        biases[l].assign(rows, 0.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                in >> weights[l][i][j];
            }
            in >> biases[l][i];
        }

    }
    in.close();
    return true;
}

void NNGo::initWeights() {
    std::mt19937 rng(std::random_device{}());

    weights.assign(HIDDEN_LAYERS + 1, {});
    biases.assign(HIDDEN_LAYERS + 1, {});

    // Hidden layers.
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inSize = layerInputSize(l);
        double stddev = std::sqrt(2.0 / inSize);
        std::normal_distribution<double> dist(0.0, stddev);

        weights[l].assign(HIDDEN_NEURONS, std::vector<double>(inSize));
        biases[l].assign(HIDDEN_NEURONS, 0.01);           // small positive bias
        for (auto& neuron : weights[l])
            for (double& w : neuron) w = dist(rng);
    }

    // Output layer.
    double stddev = std::sqrt(2.0 / HIDDEN_NEURONS);
    std::normal_distribution<double> dist(0.0, stddev);

    weights[HIDDEN_LAYERS].assign(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS));
    biases[HIDDEN_LAYERS].assign(OUTPUT_SIZE, 0.0);
    for (auto& neuron : weights[HIDDEN_LAYERS])
        for (double& w : neuron) w = dist(rng);
}

void NNGo::softmax() {
    double m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    probs.resize(OUTPUT_SIZE);
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        probs[o] = std::exp(logits[o] - m);
        sum += probs[o];
    }
    for (double& p : probs) p /= sum;
}

void NNGo::forwardPropagate(const std::vector<std::vector<std::vector<double>>>& X) {
    // Flatten and cache the input: index = c*BS*BS + r*BS + col.
    input_flat.resize(INPUT_SIZE);
    for (int c = 0; c < INPUT_CHANNELS; ++c)
        for (int r = 0; r < BOARD_SIZE; ++r)
            for (int col = 0; col < BOARD_SIZE; ++col)
                input_flat[c * BOARD_SIZE * BOARD_SIZE + r * BOARD_SIZE + col] = X[c][r][col];

    hidden.assign(HIDDEN_LAYERS, std::vector<double>(HIDDEN_NEURONS));
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inSize = layerInputSize(l);
        for (int i = 0; i < HIDDEN_NEURONS; ++i) {
            double sum = biases[l][i];
            for (int j = 0; j < inSize; ++j) {
                double prev = (l == 0) ? input_flat[j] : hidden[l - 1][j];
                sum += prev * weights[l][i][j];
            }
            hidden[l][i] = relu(sum);
        }
    }

    // Output layer -> raw logits, then softmax.
    logits.assign(OUTPUT_SIZE, 0.0);
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        double sum = biases[HIDDEN_LAYERS][o];
        for (int i = 0; i < HIDDEN_NEURONS; ++i)
            sum += hidden[HIDDEN_LAYERS - 1][i] * weights[HIDDEN_LAYERS][o][i];
        logits[o] = sum;
    }
    softmax();
}

double NNGo::crossEntropyLoss(const std::vector<double>& y_true,const std::vector<double>& y_pred) const {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        loss += -y_true[i] * std::log(std::max(y_pred[i], 1e-12));
    return loss; // sum over classes; one-hot -> -log(prob of correct move)
}

void NNGo::backwardPropagate(const std::vector<double>& y_true, double reward) {
    std::vector<std::vector<double>> delta(HIDDEN_LAYERS + 1);
    double adjustedReward = reward;

    // Output delta
    delta[HIDDEN_LAYERS].resize(OUTPUT_SIZE);
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        double d = (probs[o] - y_true[o]) * adjustedReward;
        
        // Gradient clipping for output layer
        if (d > 5.0) d = 5.0;
        if (d < -5.0) d = -5.0;
        
        delta[HIDDEN_LAYERS][o] = d;
    }

    // Hidden deltas, top-down
    for (int l = HIDDEN_LAYERS - 1; l >= 0; --l) {
        int nextSize = (l == HIDDEN_LAYERS - 1) ? OUTPUT_SIZE : HIDDEN_NEURONS;
        delta[l].assign(HIDDEN_NEURONS, 0.0);
        for (int i = 0; i < HIDDEN_NEURONS; ++i) {
            double d = 0.0;
            for (int k = 0; k < nextSize; ++k) {
                d += delta[l + 1][k] * weights[l + 1][k][i];
            }
            d *= relu_derivative(hidden[l][i]);

            // GRADIENT CLIPPING: Prevents exploding gradients / NaNs
            if (d > 5.0) d = 5.0;
            if (d < -5.0) d = -5.0;

            delta[l][i] = d;
        }
    }

    // Biases and weights update for the output layer
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        biases[HIDDEN_LAYERS][o] -= learning_rate * delta[HIDDEN_LAYERS][o];
        for (int i = 0; i < HIDDEN_NEURONS; ++i) {
            weights[HIDDEN_LAYERS][o][i] -= learning_rate * delta[HIDDEN_LAYERS][o] * hidden[HIDDEN_LAYERS - 1][i];
        }
    }

    // Biases and weights update for the hidden layers
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inSize = layerInputSize(l);
        for (int i = 0; i < HIDDEN_NEURONS; ++i) {
            biases[l][i] -= learning_rate * delta[l][i];
            for (int j = 0; j < inSize; ++j) {
                double prev = (l == 0) ? input_flat[j] : hidden[l - 1][j];
                weights[l][i][j] -= learning_rate * delta[l][i] * prev;
            }
        }
    }
}



void NNGo::train(const std::vector<std::vector<std::vector<std::vector<double>>>>& X,const std::vector<std::vector<double>>& y,int epochs) {
    for (int e = 0; e < epochs; ++e) {
        double epochLoss = 0.0;
        for (size_t n = 0; n < X.size(); ++n) {
            forwardPropagate(X[n]);
            epochLoss += crossEntropyLoss(y[n], probs);
            backwardPropagate(y[n], 1.0);
        }
        std::cout << "epoch " << e << "  avg loss " << epochLoss / X.size() << "\n";
        if (e % 100 == 0) {
            save("model_weights.txt");
        }
    }
}



void NNGo::trainOnEpisodes(const std::vector<memstep>& episodes, CellState winner, double finalMargin) {
    for (const memstep& step : episodes) {
        if (step.action < 0 || step.action >= OUTPUT_SIZE) continue;
        forwardPropagate(step.state);
        std::vector<double> y_true(OUTPUT_SIZE, 0.0);
        y_true[step.action] = 1.0;

        // base out come evalution
        double baseReward = (step.player == winner) ? 2.0 : -2.0;

        double marginBonus = (step.player == BLACK_STONE) ? finalMargin : -finalMargin;
        
        double reward = baseReward + (marginBonus * 0.05);
        
        // non linear scaling for the amount of captures made(captures)
        if(step.captsMade > 0) {
            reward += step.captsMade * 1.5;
            if (step.captsMade > 1) {
                reward += (step.captsMade * step.captsMade) * 0.8;
            }
        }
        // non linear scaling for the amount of stones lost(stones captured by the opponent)
        if(step.stonesLost > 0) {
            reward -= step.stonesLost * 1.0;
            if (step.stonesLost > 2) {
                reward -= (step.stonesLost * step.stonesLost) * 0.8;
            }
        }
        reward += step.connections * 0.05;
        reward -= 1;

        reward = std::max(-10.0, std::min(10.0, reward));
        
        backwardPropagate(y_true, reward);
    }
}
