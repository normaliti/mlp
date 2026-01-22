NAME_SPLIT = split
NAME_TRAIN = train
NAME_PREDICT = predict

CXX = c++
CXXFLAGS = -std=c++11 -Wall -Wextra -Werror -Iinclude

SRC_DIR = src
OBJ_DIR = obj

COMMON_SRCS = \
	$(SRC_DIR)/data_csv.cpp \
	$(SRC_DIR)/utils.cpp \
	$(SRC_DIR)/mlp.cpp

SPLIT_SRCS = $(SRC_DIR)/main_split.cpp
TRAIN_SRCS = $(SRC_DIR)/main_train.cpp
PREDICT_SRCS = $(SRC_DIR)/main_predict.cpp

COMMON_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))
SPLIT_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SPLIT_SRCS))
TRAIN_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TRAIN_SRCS))
PREDICT_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(PREDICT_SRCS))

all: $(NAME_SPLIT) $(NAME_TRAIN) $(NAME_PREDICT)

$(NAME_SPLIT): $(COMMON_OBJS) $(SPLIT_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(NAME_TRAIN): $(COMMON_OBJS) $(TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(NAME_PREDICT): $(COMMON_OBJS) $(PREDICT_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME_SPLIT) $(NAME_TRAIN) $(NAME_PREDICT)

re: fclean all

.PHONY: all clean fclean re
