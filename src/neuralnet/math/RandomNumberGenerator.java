package neuralnet.math;

import java.util.Random;

public class RandomNumberGenerator {

    public static long s = 0;
    public static Random r = new Random();

    public static double GenerateNext() {
        return r.nextDouble();
    }

    public static void setSeed(long seed) {
        s = seed;
        r.setSeed(seed);
    }
}

