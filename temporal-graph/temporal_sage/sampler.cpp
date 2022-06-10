#include <sstream>
#include <cassert>
#include <imports.hpp>

using imports::log_info;
using imports::log_error;

template<typename T>
void ugly_concat_impl(std::stringstream& ss, const T& t) { ss << t; }

template<typename T, typename ... Args>
void ugly_concat_impl(std::stringstream& ss, const T& t, Args ... args) { ss << t; ugly_concat_impl(ss, args...); }

template<typename ... Args>
std::string ugly_concat(Args ... args) {
    std::stringstream ss;
    ugly_concat_impl(ss, args...);
    return ss.str();
}

imports::item_param __support_item_types[] = {
    imports::item_param { "bool", false },
    imports::item_param { "int32", 1 },
    imports::item_param { "int64", 2LL },
    imports::item_param { "float32", 3.0f },
    imports::item_param { "float64", 4.0 },
    imports::item_param { "string", "5.0" },
};


void test_query_nodes(imports::storage& store, auto a) {
    // age 是int64_t类型
    auto table = imports::data_frame::open("query_nodes(1005)", {
        { "author_id", -1},
        { "label", -1},
    }).unwrap();

    auto r = store.query_nodes(a, "author", {"author_id", "label"}).unwrap();

    assert(r.view()["author.author_id"].unwrap().is_i64()); // 实际类型都为i64
    assert(r.view()["author.label"].expect("no author.label").is_i64()); // 使用expect解包以获得更明确的错误信息

    auto author_id = r.view()["author.author_id"].unwrap().to_i32(); // 强制转成i32
    auto label = r.view()["author.label"].unwrap().to_i32();
    table.push({
        {"author_id", author_id},
        {"label", label},
    });
}

void test_query_neighbors(imports::storage& store, auto a) {
    auto table = imports::data_frame::open("test_query_neighbors(1005)", {
        {"target", -1LL},
        {"time_stamp", -1LL},
    }).unwrap();

    auto [dst, attr] = store.query_neighbors(a, "coauthor", {"time_stamp"}).unwrap();

    assert(attr.view()["coauthor.time_stamp"].unwrap().is_i64()); // 实际类型为i64

    auto n = dst.view().size();
    for (size_t i = 0; i < n; i++) {
        imports::item_param data[] = {
            {"target", dst.view().as_i64()[i]},
            {"time_stamp", attr.view()["coauthor.time_stamp"].unwrap().as_i64()[i]}, // 这里不发生强制类型转换
        };
        table.push(data);
    }
}

int main(int argc, char* argv[]) {

    if (argc != 0) {
        // 获得整数类型的参数
        auto a = imports::txt2i32(argv[0]);
        imports::log_info<"args: {}">(a);

        // 如果需要在模板函数里使用，可以这样写
        auto b = imports::__from_txt<imports::i32>(argv[0]);
        imports::log_info<"args: {}">(b);
    }
    auto a = imports::txt2i32(argv[0]);
    auto store = imports::storage::open().unwrap();
    // test_choice_nodes(store);
    // test_query_nodes(store, a);
    test_query_neighbors(store, a);
    return 0;
}