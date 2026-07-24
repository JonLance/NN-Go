// changed to 3d vectors so it does not see the game in a strait line but i did not make a dicition tree so no plan in the end sad
#include <cstddef>
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

struct memstep {
    std::vector<std::vector<std::vector<double>>> state; // [channel][row][col]
    int action;
    CellState player;
    int captsMade;
    int stonesLost;
    int connections;
    std::vector<double> territoryTarget;
    std::vector<double> legal_moves;
};


class NNGo {
public:
    static constexpr int BOARD_SIZE = 19;
    static constexpr int INPUT_CHANNELS = 5;
    static constexpr int NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE;
    static constexpr int OUTPUT_SIZE = NUM_ACTIONS + 1; // 361 moves
    
    static constexpr int HIDDEN_LAYERS = 5;
    static constexpr int HIDDEN_CHANNELS = 32; // 32 feature maps instead of flat neurons

    double learning_rate = 0.0005;

    // 3D Spatial States cached during forward pass
    std::vector<std::vector<std::vector<double>>> input_cache; // [channel][row][col]
    std::vector<std::vector<std::vector<std::vector<double>>>> hidden; // [layer][channel][row][col]
    std::vector<double> logits; // OUTPUT_SIZE (pre-softmax)
    std::vector<double> probs;  // OUTPUT_SIZE (post-softmax)
    
    // territory stuff (do not know what to call it)
    std::vector<double> weightsTerritory; // [hidden_channels]
    double biasTerr = 0.0;
    std::vector<double> territory; // [output_size]
    double territoryW = 1.2;
    std::vector<std::vector<std::vector<std::vector<double>>>> deltaHidden;
    
    
    // Hidden layers 
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> weights;
    std::vector<std::vector<double>> biases; // [layer][channel]
    const double scale = 1.0 / (BOARD_SIZE * BOARD_SIZE);
    // Output layer weights (Fully Connected from last 3D hidden layer to 361 outputs)
    // [output_move][channel][row][col]
    std::vector<std::vector<std::vector<std::vector<double>>>> weights_out;
    std::vector<double> biases_out; // [output_move]



    void initWeights();
    void forwardPropagate(const std::vector<std::vector<std::vector<double>>>& X, const std::vector<bool>& legal_moves = std::vector<bool>(OUTPUT_SIZE, true));
    void backwardPropagate(const std::vector<double>& y_true, double reward,const std::vector<double> &territoryTarget = {});
    void train(const std::vector<std::vector<std::vector<std::vector<double>>>>& X, const std::vector<std::vector<double>>& y, int epochs);
    void trainOnEpisodes(const std::vector<memstep>& episodes, CellState winner, double finalMargin);
    double crossEntropyLoss(const std::vector<double>& y_true, const std::vector<double>& y_pred) const;
    bool save(const std::string& filename);
    bool load(const std::string& filename);
    void forwardteritory();
    void trainingintime(int reward, int capts, int teritory);
    void softmax(const std::vector<bool>& legal_moves);

private:
    static inline double relu(double x) { return x > 0 ? x : 0.0; }
    static inline double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
};
#endif
bool NNGo::save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << HIDDEN_LAYERS << " " << HIDDEN_CHANNELS << std::endl;

    // Save convolution hidden layers
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inChannels = (l == 0) ? INPUT_CHANNELS : HIDDEN_CHANNELS;
        for (int out_c = 0; out_c < HIDDEN_CHANNELS; ++out_c) {
            for (int in_c = 0; in_c < inChannels; ++in_c) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out << weights[l][out_c][in_c][r][c] << " ";
                    }
                }
            }
            out << biases[l][out_c] << " ";
        }
        out << std::endl;
    }

    // Save output layer
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    out << weights_out[o][ch][r][c] << " ";
                }
            }
        }
        out << biases_out[o] << " ";
    }
    out << std::endl;
    for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) out << weightsTerritory[ch] << " ";
    out << biasTerr << std::endl;
    out.close();
    return true;
}

bool NNGo::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    int file_layers, file_channels;
    if (!(in >> file_layers >> file_channels)) return false;
    if (file_layers != HIDDEN_LAYERS || file_channels != HIDDEN_CHANNELS) return false;

    weights.resize(HIDDEN_LAYERS);
    biases.assign(HIDDEN_LAYERS, std::vector<double>(HIDDEN_CHANNELS));

    // Load hidden conv weights
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inChannels = (l == 0) ? INPUT_CHANNELS : HIDDEN_CHANNELS;
        weights[l].assign(HIDDEN_CHANNELS, std::vector<std::vector<std::vector<double>>>(
            inChannels, std::vector<std::vector<double>>(3, std::vector<double>(3))));
        
        for (int out_c = 0; out_c < HIDDEN_CHANNELS; ++out_c) {
            for (int in_c = 0; in_c < inChannels; ++in_c) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        in >> weights[l][out_c][in_c][r][c];
                    }
                }
            }
            in >> biases[l][out_c];
        }
    }

    // Load output weights
    weights_out.assign(OUTPUT_SIZE, std::vector<std::vector<std::vector<double>>>(HIDDEN_CHANNELS, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE))));
    biases_out.assign(OUTPUT_SIZE, 0.0);
    
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    in >> weights_out[o][ch][r][c];
                }
            }
        }
        
        in >> biases_out[o];
    }
    weightsTerritory.assign(HIDDEN_CHANNELS, 0.0);
    for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) in >> weightsTerritory[ch];
    in >> biasTerr;
    if (!in) {                       // old file without the territory head
        double sd = std::sqrt(2.0 / HIDDEN_CHANNELS);
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> d(0.0, sd);
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) weightsTerritory[ch] = d(rng);
        biasTerr = 0.0;
    }
    
    in.close();
    return true;
}

void NNGo::initWeights() {
    std::mt19937 rng(std::random_device{}());

    weights.resize(HIDDEN_LAYERS);
    biases.assign(HIDDEN_LAYERS, std::vector<double>(HIDDEN_CHANNELS, 0.01));

    // Initialize Hidden Conv Layers
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inChannels = (l == 0) ? INPUT_CHANNELS : HIDDEN_CHANNELS;
        weightsTerritory.assign(HIDDEN_CHANNELS, 0.0);
        
        double stddev = std::sqrt(2.0 / (inChannels * 3 * 3));
        std::normal_distribution<double> dist(0.0, stddev);

        weights[l].assign(HIDDEN_CHANNELS, std::vector<std::vector<std::vector<double>>>(inChannels, std::vector<std::vector<double>>(3, std::vector<double>(3))));
        
        for (int out_c = 0; out_c < HIDDEN_CHANNELS; ++out_c) {
            for (int in_c = 0; in_c < inChannels; ++in_c) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        weights[l][out_c][in_c][r][c] = dist(rng);
                    }
                }
            }
        }
    }

    // Initialize Output Layer Weights
    double stddev_out = std::sqrt(2.0 / (HIDDEN_CHANNELS * BOARD_SIZE * BOARD_SIZE));
    std::normal_distribution<double> dist_out(0.0, stddev_out);

    weights_out.assign(OUTPUT_SIZE, std::vector<std::vector<std::vector<double>>>(HIDDEN_CHANNELS, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE))));
    biases_out.assign(OUTPUT_SIZE, 0.0);

    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    weights_out[o][ch][r][c] = dist_out(rng);
                }
            }
        }
    }
    double stddev_territory = std::sqrt(2.0 / HIDDEN_CHANNELS);
    std::normal_distribution<double> dist_territory(0.0, stddev_territory);

    for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
        weightsTerritory[ch] = dist_territory(rng);
    }
    biasTerr = 0.0;
}

void NNGo::softmax(const std::vector<bool>& legal_moves) {
    // 1. Resize/allocate probs so operator[] stops crashing
    probs.assign(OUTPUT_SIZE, 0.0);

    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        if (!legal_moves[o]) {
            logits[o] = -1e9;
        }
    }

    double m = -1e9;
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        if (legal_moves[o] && logits[o] > m) {
            m = logits[o];
        }
    }

    double sum = 0.0;
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        if (legal_moves[o]) {
            probs[o] = std::exp(logits[o] - m);
            sum += probs[o];
        } else {
            probs[o] = 0.0;
        }
    }

    if (sum > 0.0 && std::isnan(sum)) {
        for (double& p : probs) {
            p /= sum;
        }
    }
    else {
        // Fallback: Uniform distribution over legal moves
        int legal_count = 0;
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            if (legal_moves[o]) legal_count++;
        }
        
        if (legal_count > 0) {
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                if (legal_moves[o]) legal_count++;
            }
            
            if (legal_count > 0) {
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    probs[o] = legal_moves[o] ? (1.0 / legal_count) : 0.0;
                }
            } else {
                // Default to pass if no legal moves exist
                probs[361] = 1.0; 
            } 
        }
    }
}

void NNGo::forwardPropagate(const std::vector<std::vector<std::vector<double>>>& X, const std::vector<bool>& legal_moves) {
    input_cache = X; 

    hidden.assign(HIDDEN_LAYERS, std::vector<std::vector<std::vector<double>>>(
        HIDDEN_CHANNELS, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0))));

    // Hidden Spatial Layers Loop
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inChannels = (l == 0) ? INPUT_CHANNELS : HIDDEN_CHANNELS;
        
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    double sum = biases[l][ch];
                    
                    for (int inch = 0; inch < inChannels; ++inch) {
                        for (int dr = -1; dr <= 1; ++dr) {
                            for (int dc = -1; dc <= 1; ++dc) {
                                int nr = r + dr;
                                int nc = c + dc;
                                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                                    double prev_val = (l == 0) ? input_cache[inch][nr][nc] : hidden[l-1][inch][nr][nc];
                                    sum += prev_val * weights[l][ch][inch][dr+1][dc+1];
                                }
                            }
                        }
                    }
                    if (l > 0) {
                        sum += hidden[l-1][ch][r][c]; // Skip connection
                    }
                    hidden[l][ch][r][c] = relu(sum);
                }
            }
        }
    }

    // Output Layer -> 361 Board Logits
    logits.assign(OUTPUT_SIZE, 0.0);
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        double sum = biases_out[o];
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    sum += hidden[HIDDEN_LAYERS-1][ch][r][c] * weights_out[o][ch][r][c];
                }
            }
        }
        logits[o] = sum;
    }
    
    softmax(legal_moves);
    forwardteritory(); 
}

void NNGo::forwardteritory() {
    territory.assign(NUM_ACTIONS, 0.0);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            double sum = biasTerr;
            for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
                sum += hidden[HIDDEN_LAYERS-1][ch][r][c] * weightsTerritory[ch];
            }
            territory[r * BOARD_SIZE + c] = std::tanh(sum);
        }
    }
}



double NNGo::crossEntropyLoss(const std::vector<double>& y_true, const std::vector<double>& y_pred) const {
    double loss = 0.0;
    // loss calculation the true labels and predicted probabilities differences
    for (size_t i = 0; i < y_true.size(); ++i)
        loss += -y_true[i] * std::log(std::max(y_pred[i], 1e-12));
    return loss;
}



void NNGo::backwardPropagate(const std::vector<double>& y_true, double reward, const std::vector<double>& territoryTarget) {
    // Output Layer Deltas
    std::vector<double> delta_out(OUTPUT_SIZE);
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        if (y_true[o] > 0) {
            delta_out[o] = (probs[o] - y_true[o]) * reward;
        }
        else {
            delta_out[o] = probs[o] * reward;
        }
        delta_out[o] = std::max(std::min(delta_out[o], 1.0), -1.0);
    }
    

    bool useTerritory = (territoryTarget.size() == (size_t)NUM_ACTIONS);
    std::vector<double> delta_territory(NUM_ACTIONS);
    if (useTerritory) {
        for (int o = 0; o < NUM_ACTIONS; ++o) {
            double t = territory[o];
            double d = 2.0 * (t - territoryTarget[o]) * (1.0 - t * t) * territoryW;
            delta_territory[o] = std::max(std::min(d, 2.0), -2.0);
        }
    }

    // Spatial Hidden Layers Deltas allocation
    std::vector<std::vector<std::vector<std::vector<double>>>> delta_hidden(HIDDEN_LAYERS, std::vector<std::vector<std::vector<double>>>(HIDDEN_CHANNELS, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0))));

    // Backpropagation to the last hidden layer
    for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                double d = 0.0;
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    d += delta_out[o] * weights_out[o][ch][r][c];
                }
                if (useTerritory) {
                    d += delta_territory[r * BOARD_SIZE + c] * weightsTerritory[ch];
                }
                d *= relu_derivative(hidden[HIDDEN_LAYERS-1][ch][r][c]);
                if (d > 5.0) d = 5.0;
                if (d < -5.0) d = -5.0;
                delta_hidden[HIDDEN_LAYERS-1][ch][r][c] = d;
            }
        }
    }

    // Backpropagation down through spatial hidden layers
    for (int l = HIDDEN_LAYERS - 2; l >= 0; --l) {
        for (int inch = 0; inch < HIDDEN_CHANNELS; ++inch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    double d = 0.0;

                    // Convolution path gradient
                    for (int outch = 0; outch < HIDDEN_CHANNELS; ++outch) {
                        for (int dr = -1; dr <= 1; ++dr) {
                            for (int dc = -1; dc <= 1; ++dc) {
                                int nr = r - dr; 
                                int nc = c - dc;
                                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                                    d += delta_hidden[l+1][outch][nr][nc] * weights[l+1][outch][inch][dr+1][dc+1];
                                }
                            }
                        }
                    }

                    d += delta_hidden[l+1][inch][r][c]; // ✅ Fixed typo to use local delta_hidden
                    d *= relu_derivative(hidden[l][inch][r][c]);

                    if (d > 5.0) d = 5.0;
                    if (d < -5.0) d = -5.0;
                    delta_hidden[l][inch][r][c] = d;
                }
            }
        }
    }

    // Output Layer Weights & Biases
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        biases_out[o] -= learning_rate * delta_out[o];
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    double scale_factor = 1.0 / (HIDDEN_CHANNELS * BOARD_SIZE * BOARD_SIZE);
                    weights_out[o][ch][r][c] -= learning_rate * delta_out[o] * hidden[HIDDEN_LAYERS-1][ch][r][c] * scale_factor;
                }
            }
        }
    }

    if (useTerritory) {
        std::vector<double> territory_grad(HIDDEN_CHANNELS, 0.0);
        double territory_bias_grad = 0.0;
        for(int r = 0; r < BOARD_SIZE; ++r) {
            for(int c = 0; c < BOARD_SIZE; ++c) {
                double d = delta_territory[r * BOARD_SIZE + c];
                for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
                    territory_grad[ch] += d * hidden[HIDDEN_LAYERS-1][ch][r][c];
                }
                territory_bias_grad += d;
            }
        }
        for (int w = 0; w < HIDDEN_CHANNELS; ++w) {
            weightsTerritory[w] -= learning_rate * territory_grad[w];
        }
        biasTerr -= learning_rate * territory_bias_grad;
    }

    // Hidden Convolutional Weights & Biases
    for (int l = 0; l < HIDDEN_LAYERS; ++l) {
        int inChannels = (l == 0) ? INPUT_CHANNELS : HIDDEN_CHANNELS;
        for (int ch = 0; ch < HIDDEN_CHANNELS; ++ch) {
            double bias_grad = 0.0;
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    bias_grad += delta_hidden[l][ch][r][c];
                }
            }
            biases[l][ch] -= learning_rate * bias_grad * scale;

            for (int inch = 0; inch < inChannels; ++inch) {
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        double weight_grad = 0.0;
                        for (int r = 0; r < BOARD_SIZE; ++r) {
                            for (int c = 0; c < BOARD_SIZE; ++c) {
                                int nr = r + dr;
                                int nc = c + dc;
                                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                                    double input_val = (l == 0) ? input_cache[inch][nr][nc] : hidden[l-1][inch][nr][nc];
                                    weight_grad += delta_hidden[l][ch][r][c] * input_val;
                                }
                            }
                        }
                        weights[l][ch][inch][dr+1][dc+1] -= learning_rate * weight_grad * scale;
                    }
                }
            }
        }
    }
}



void NNGo::train(const std::vector<std::vector<std::vector<std::vector<double>>>>& X, const std::vector<std::vector<double>>& y, int epochs) {
    for (int e = 0; e < epochs; ++e) {
        double epochLoss = 0.0;
        for (size_t n = 0; n < X.size(); ++n) {
            forwardPropagate(X[n]);
            epochLoss += crossEntropyLoss(y[n], probs);
            backwardPropagate(y[n], 1.0, {});
        }
        std::cout << "Epoch " << e << " | Avg Loss: " << epochLoss / X.size() << "\n";
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

        double baseReward = (step.player == winner) ? 2.0 : -2.0;
        double marginBonus = (step.player == BLACK_STONE) ? finalMargin : -finalMargin;
        double reward = baseReward + (marginBonus * 0.05);
        
        

        reward = std::max(-2.0, std::min(2.0, reward));
        backwardPropagate(y_true, reward, step.territoryTarget);
    }
}

