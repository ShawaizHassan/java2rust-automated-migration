import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class TheMonster {
	public static void main(String[] args) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		String[] p = in.readLine().split(" ");
		int a = Integer.parseInt(p[0]);
		long b = Long.parseLong(p[1]);
		
		boolean aa = a % 2 == 0;
		boolean bb = b % 2 == 0;
		
		String[] q = in.readLine().split(" ");
		int c = Integer.parseInt(q[0]);
		long d = Long.parseLong(q[1]);
		
		boolean cc = c % 2 == 0;
		boolean dd = d % 2 == 0;
		
		if(!((!bb && aa && dd && cc)||(!dd&&cc&&bb&&aa))) {
			while(b != d) {
				if(b < d) {
					b+=a;
				} else {
					d+=c;
				}
				if(d > 10000000) {
					b = -1;
					d = -1;
				}
			}
			System.out.println(b);
		} else {
			System.out.println(-1);
		}
	}
}