# === Configurable Variables ===
CXX = icpx
SRC = buffered_copies.cpp
TARGET = buffered_copies
CXXFLAGS = -fsycl -qopenmp -O2 -std=c++20

# === Default Target ===
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# === Cleanup ===
clean:
	rm -f $(TARGET)

