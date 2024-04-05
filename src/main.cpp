#include <iostream>

int main(int argc, char **argv)
{
    std::cout << "There are " << argc << " arguments." << std::endl;

    std::cout << "They are:" << std::endl;

    for (int idx = 0; idx < argc; ++idx)
    {
        std::cout << argv[idx] << std::endl;
    }
    
    return 0;
}