import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.Arrays;

import java.util.StringTokenizer;


public class Codeforces {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solve = new TaskA();
        solve.solve(1, in, out);
        out.close();
    }
    static class TaskA {
        public void solve(int testNumber, InputReader in, PrintWriter out) {
           int n = in.nextInt();
           int m = in.nextInt();
           
           int arr[] = new int[m];
           
           
           for(int i = 0; i < m; ++i) {
               arr[i] = in.nextInt();
           }
           
           Arrays.sort(arr);
           
           int minimum = Integer.MAX_VALUE;
           
           if(m > n) {
               n--;
                for(int i = 0; i < m - n; ++i) {
                    minimum = Math.min(minimum, arr[i + n] - arr[i]);
                }
           } else if(m==n) {
               minimum = Math.min(minimum, arr[n-1] - arr[0]);
           } else {
                for(int j = 0; j < n-1; ++j) {
                    minimum = Math.min(minimum, arr[j + n-1] - arr[j]);
                }
           }
           
           
           
           out.println(minimum);
           
        }
        
        private static String reverse(String word) {
            String out = "";
            for(int i = word.length() -1; i >= 0; i--) {
                out += word.charAt(i);
            }
            return out;
        }
    }
    static class InputReader {
        public BufferedReader reader;
        public StringTokenizer tokenizer;
 
        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }
 
        public String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }
 
        public int nextInt() {
            return Integer.parseInt(next());
        }
        
        public double nextDouble() {
            return Double.parseDouble(next());
        }
 
    }
}