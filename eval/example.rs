use std::io;

fn main() {
    let mut inc = 0;
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let mut iter = input.split_whitespace();
    let n: usize = iter.next().unwrap().parse().expect("Please type a number!");
    let k: usize = iter.next().unwrap().parse().expect("Please type a number!");
    let mut a = Vec::new();
    for _ in 0..n {
        input.clear();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        let num: i32 = input.trim().parse().expect("Please type a number!");
        a.push(num);
    }
    for i in 0..n {
        if a[i] >= a[k - 1] && a[i] > 0 {
            inc += 1;
        }
    }
    println!("{}", inc);
}