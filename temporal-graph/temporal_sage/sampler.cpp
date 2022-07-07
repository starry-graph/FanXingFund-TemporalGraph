#include <imports.hpp>

void query_neighbors(imports::storage& store, imports::i64 s) {
    auto table = imports::data_frame::open("neighbors", {
        {"target", -1LL},
        {"time_stamp", -1LL},
    }).expect("opened table");

    auto [dst, attr] = store.query_neighbors(s, "coauthor", {"time_stamp"}).expect("query coauthor");

    auto target = dst.view().to_i64();
    auto time_stamp = attr.view()["coauthor.time_stamp"].expect("query timestamp").to_i64();

    for (size_t i = 0; i < target.size(); i++) {
        table.push({
	    {"target", target[i]},
	    {"time_stamp", time_stamp[i]},
	});
    }
}

int main(int argc, char* argv[]) {
    if (argc != 0) {
        auto s = imports::txt2i64(argv[0]);
        auto store = imports::storage::open().expect("opened storage");
        query_neighbors(store, s);
    }
    return 0;
}
