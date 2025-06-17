# Compiler and Options
CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17

# Common
COMMON_SRC = functions.cpp NeuralNetwork.cpp
COMMON_OBJ = $(COMMON_SRC:.cpp=.o)

# Executables
TRAIN = train
TEST = test

# Default : all
all: $(TRAIN) $(TEST)

# Compile train
$(TRAIN): train.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile test
$(TEST): test.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile c++ to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f *.o $(TRAIN) $(TEST)

